import os
import time
import copy
import math
import random
from statistics import NormalDist

import torch
import torch.optim as optim

from cvrp_model import CVRPModel
from cvrp_env import CVRPenv


# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# PATHS
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# checkpoint cũ để warm-start
WARM_START_CKPT = os.path.join(BASE_DIR, "checkpoints_neural_fill", "model_epoch_30.pt")

# nơi lưu checkpoint mới
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints_neural_rollout")
os.makedirs(SAVE_DIR, exist_ok=True)


# =====================
# HYPERPARAMS
# =====================
EPOCHS = 30
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LR = 1e-4
BATCHES_PER_EPOCH = 1000

# giữ benchmark của bạn
TRAIN_NODE_SIZES = [100, 120, 150]

# baseline update settings
BASELINE_ALPHA = 0.05
BASELINE_EVAL_SAMPLES = 2048   # test nhanh trước; muốn sát paper hơn có thể tăng 10000
GRAD_CLIP_NORM = 1.0

# capacity map cho mixed sizes
CAPACITY_BY_SIZE = {
    100: 50.0,
    120: 50.0,
    150: 50.0,
}

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================
# MODEL
# =====================
model = CVRPModel(
    embedding_dim=128,
    num_heads=8,
    num_layers=3
).to(device)

# warm-start từ checkpoint cũ
if os.path.exists(WARM_START_CKPT):
    ckpt = torch.load(WARM_START_CKPT, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print(f"[INFO] Warm-start loaded from: {WARM_START_CKPT}")
else:
    print(f"[WARN] Warm-start checkpoint not found: {WARM_START_CKPT}")
    print("[WARN] Training will start from scratch.")

# baseline model = best model so far
baseline_model = copy.deepcopy(model).eval()

# reset optimizer mới
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Device: {device}")
print(f"Train node sizes: {TRAIN_NODE_SIZES}")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Batches/epoch: {BATCHES_PER_EPOCH}")
print(f"Eval samples per epoch: {BASELINE_EVAL_SAMPLES}")


# =====================
# HELPERS
# =====================
def get_capacity(num_nodes: int) -> float:
    return CAPACITY_BY_SIZE.get(num_nodes, 50.0)


def make_shared_batch(num_nodes: int, batch_size: int):
    """
    Tạo cùng một batch instances để current model và baseline model
    đánh giá trên đúng cùng dữ liệu.
    """
    capacity = get_capacity(num_nodes)

    env = CVRPenv(num_nodes=num_nodes, capacity=capacity, device=device)
    env.reset(batch_size=batch_size)

    locs = env.locs.clone()
    demands = env.demands.clone()
    return locs, demands, capacity


def run_model_on_batch(model_to_eval, locs, demands, num_nodes, capacity, decode_type):
    """
    Chạy model trên batch dữ liệu cố định.
    Return:
        rewards: [B] (âm)
        log_probs: [B] hoặc None nếu greedy mà không cần dùng
    """
    env = CVRPenv(num_nodes=num_nodes, capacity=capacity, device=device)
    env.reset(batch_size=locs.size(0), locs=locs, demands=demands)

    rewards, log_probs = model_to_eval(env, decode_type=decode_type)
    return rewards, log_probs


@torch.no_grad()
def paired_t_test_one_sided(candidate_costs, baseline_costs):
    """
    H1: candidate tốt hơn baseline, tức candidate_cost < baseline_cost
    Dùng d = baseline - candidate; mean(d) > 0 là tốt.
    """
    d = baseline_costs - candidate_costs
    n = d.numel()

    mean_d = d.mean().item()
    std_d = d.std(unbiased=True).item()

    if std_d < 1e-12:
        return 0.0 if mean_d > 0 else 1.0

    t_stat = mean_d / (std_d / math.sqrt(n))
    p_value = 1.0 - NormalDist().cdf(t_stat)
    return p_value


@torch.no_grad()
def evaluate_for_baseline_update(candidate_model, baseline_model, num_samples=BASELINE_EVAL_SAMPLES):
    """
    Đánh giá candidate vs baseline trên mixed sizes [100,120,150].
    """
    candidate_all = []
    baseline_all = []

    remaining = num_samples
    while remaining > 0:
        bs = min(EVAL_BATCH_SIZE, remaining)
        num_nodes = random.choice(TRAIN_NODE_SIZES)

        locs, demands, capacity = make_shared_batch(num_nodes, bs)

        cand_rewards, _ = run_model_on_batch(
            candidate_model, locs, demands, num_nodes, capacity, decode_type="greedy"
        )
        base_rewards, _ = run_model_on_batch(
            baseline_model, locs, demands, num_nodes, capacity, decode_type="greedy"
        )

        candidate_all.append(-cand_rewards)  # cost dương
        baseline_all.append(-base_rewards)

        remaining -= bs

    candidate_costs = torch.cat(candidate_all, dim=0)
    baseline_costs = torch.cat(baseline_all, dim=0)

    cand_mean = candidate_costs.mean().item()
    base_mean = baseline_costs.mean().item()
    p_value = paired_t_test_one_sided(candidate_costs, baseline_costs)

    return cand_mean, base_mean, p_value


def save_checkpoint(path, epoch, model, optimizer, baseline_model, stats):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "baseline_model_state_dict": baseline_model.state_dict(),
            "stats": stats,
            "train_node_sizes": TRAIN_NODE_SIZES,
        },
        path,
    )


# =====================
# TRAIN LOOP
# =====================
for epoch in range(1, EPOCHS + 1):
    model.train()
    baseline_model.eval()

    start_time = time.time()

    epoch_loss = 0.0
    epoch_train_cost = 0.0
    epoch_baseline_cost = 0.0

    for batch_id in range(1, BATCHES_PER_EPOCH + 1):
        num_nodes = random.choice(TRAIN_NODE_SIZES)
        batch_size = BATCH_SIZE
        locs, demands, capacity = make_shared_batch(num_nodes, batch_size)

        # current model: sampling rollout
        rewards_cur, log_probs = run_model_on_batch(
            model, locs, demands, num_nodes, capacity, decode_type="sampling"
        )
        sample_costs = -rewards_cur  # cost dương

        # baseline:
        # epoch 1 dùng warmup mean baseline giống tinh thần paper
        if epoch == 1:
            baseline_cost = sample_costs.mean().expand_as(sample_costs)
        else:
            with torch.no_grad():
                rewards_base, _ = run_model_on_batch(
                    baseline_model, locs, demands, num_nodes, capacity, decode_type="greedy"
                )
                baseline_cost = -rewards_base

        # REINFORCE với cost:
        # loss = E[(cost - baseline) * log_prob]
        advantage = sample_costs - baseline_cost
        loss = (advantage.detach() * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        batch_train_cost = sample_costs.mean().item()
        batch_baseline_cost = baseline_cost.mean().item()

        epoch_loss += loss.item()
        epoch_train_cost += batch_train_cost
        epoch_baseline_cost += batch_baseline_cost

        if batch_id % 100 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Batch {batch_id:04d}/{BATCHES_PER_EPOCH} | "
                f"nodes={num_nodes} | "
                f"avg_train_cost={epoch_train_cost / batch_id:.4f} | "
                f"avg_baseline_cost={epoch_baseline_cost / batch_id:.4f} | "
                f"avg_loss={epoch_loss / batch_id:.4f}"
            )

    # =====================
    # END OF EPOCH: baseline update test
    # =====================
    model.eval()
    cand_mean, base_mean, p_value = evaluate_for_baseline_update(
        candidate_model=model,
        baseline_model=baseline_model,
        num_samples=BASELINE_EVAL_SAMPLES,
    )

    baseline_updated = False
    if cand_mean < base_mean and p_value < BASELINE_ALPHA:
        baseline_model = copy.deepcopy(model).eval()
        baseline_updated = True

    epoch_time = time.time() - start_time
    avg_epoch_loss = epoch_loss / BATCHES_PER_EPOCH
    avg_epoch_train_cost = epoch_train_cost / BATCHES_PER_EPOCH
    avg_epoch_baseline_cost = epoch_baseline_cost / BATCHES_PER_EPOCH

    print(
        f"\nEpoch {epoch:02d} finished | "
        f"Avg Train Cost: {avg_epoch_train_cost:.4f} | "
        f"Avg Baseline Cost: {avg_epoch_baseline_cost:.4f} | "
        f"Avg Loss: {avg_epoch_loss:.4f} | "
        f"Eval Cand: {cand_mean:.4f} | "
        f"Eval Base: {base_mean:.4f} | "
        f"p-value: {p_value:.6f} | "
        f"baseline_updated={baseline_updated} | "
        f"Time: {epoch_time:.2f}s\n"
    )

    ckpt_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pt")
    save_checkpoint(
        path=ckpt_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        baseline_model=baseline_model,
        stats={
            "avg_loss": avg_epoch_loss,
            "avg_train_cost": avg_epoch_train_cost,
            "avg_baseline_cost": avg_epoch_baseline_cost,
            "eval_candidate_cost": cand_mean,
            "eval_baseline_cost": base_mean,
            "p_value": p_value,
            "baseline_updated": baseline_updated,
        },
    )

# save final
final_path = os.path.join(SAVE_DIR, "model_final.pt")
save_checkpoint(
    path=final_path,
    epoch=EPOCHS,
    model=model,
    optimizer=optimizer,
    baseline_model=baseline_model,
    stats={"message": "final checkpoint"},
)

print(f"Training finished. Final model saved to: {final_path}")