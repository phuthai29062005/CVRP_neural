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


# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# PATHS
# =========================================================
# Nếu train.py nằm trong thư mục Train_Neural/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Train từ 0
WARM_START = False
WARM_START_CKPT = os.path.join(BASE_DIR, "checkpoints_neural_fill", "model_epoch_30.pt")

SAVE_DIR = os.path.join(BASE_DIR, "checkpoints_neural_rollout_curriculum")
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================
TRAIN_NODE_SIZES = [100, 120, 150]

EPOCHS = 30
BATCHES_PER_EPOCH = 600
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128

# From scratch nên để 5e-5 hoặc 1e-4. Bản này chọn an toàn hơn.
LR = 5e-5
GRAD_CLIP = 1.0

BASELINE_EVAL_SAMPLES = 2048
BASELINE_ALPHA = 0.05

# exponential baseline cho epoch 1
EXP_BETA = 0.9

CAPACITY = {
    100: 50.0,
    120: 50.0,
    150: 50.0,
}

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# UTILS
# =========================================================
def sample_nodes() -> int:
    return random.choice(TRAIN_NODE_SIZES)


def get_vehicle_penalty(epoch: int) -> float:
    """
    Curriculum penalty:
    epoch  1-10  -> 200
    epoch 11-20  -> 500
    epoch 21-30+ -> 1000
    """
    if epoch <= 10:
        return 200.0
    elif epoch <= 20:
        return 500.0
    else:
        return 1000.0


def make_batch(num_nodes: int, batch_size: int, vehicle_penalty: float):
    env = CVRPenv(
        num_nodes=num_nodes,
        capacity=CAPACITY[num_nodes],
        device=device,
        vehicle_penalty=vehicle_penalty,
    )
    env.reset(batch_size=batch_size)
    return env.locs.clone(), env.demands.clone()


def run_model(
    model,
    locs,
    demands,
    num_nodes: int,
    decode_type: str,
    vehicle_penalty: float,
):
    env = CVRPenv(
        num_nodes=num_nodes,
        capacity=CAPACITY[num_nodes],
        device=device,
        vehicle_penalty=vehicle_penalty,
    )
    env.reset(batch_size=locs.size(0), locs=locs, demands=demands)

    rewards, log_probs = model(env, decode_type=decode_type)

    # objective = vehicle_penalty * routes + distance
    cost = -rewards
    route_count = env.route_count.float()
    total_distance = env.total_distance

    return cost, log_probs, route_count, total_distance


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


@torch.no_grad()
def paired_t_test_one_sided(candidate_costs, baseline_costs):
    """
    H1: candidate tốt hơn baseline <=> candidate_cost < baseline_cost
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
def evaluate(candidate_model, baseline_model, vehicle_penalty: float):
    cand_all = []
    base_all = []

    remaining = BASELINE_EVAL_SAMPLES
    while remaining > 0:
        bs = min(EVAL_BATCH_SIZE, remaining)
        n = sample_nodes()

        locs, demands = make_batch(n, bs, vehicle_penalty)

        cand_cost, _, _, _ = run_model(
            candidate_model, locs, demands, n, "greedy", vehicle_penalty
        )
        base_cost, _, _, _ = run_model(
            baseline_model, locs, demands, n, "greedy", vehicle_penalty
        )

        cand_all.append(cand_cost)
        base_all.append(base_cost)

        remaining -= bs

    cand = torch.cat(cand_all, dim=0)
    base = torch.cat(base_all, dim=0)

    cand_mean = cand.mean().item()
    base_mean = base.mean().item()
    p_value = paired_t_test_one_sided(cand, base)

    return cand_mean, base_mean, p_value


# =========================================================
# MODEL
# =========================================================
model = CVRPModel(
    embedding_dim=128,
    num_heads=8,
    num_layers=3
).to(device)

if WARM_START and os.path.exists(WARM_START_CKPT):
    ckpt = torch.load(WARM_START_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print(f"[INFO] Warm-start loaded from: {WARM_START_CKPT}")
elif WARM_START:
    print(f"[WARN] Warm-start checkpoint not found: {WARM_START_CKPT}")
    print("[WARN] Training will start from scratch.")
else:
    print("[INFO] Training from scratch.")

baseline_model = copy.deepcopy(model).eval()
for p in baseline_model.parameters():
    p.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Device: {device}")
print(f"Train node sizes: {TRAIN_NODE_SIZES}")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Batches/epoch: {BATCHES_PER_EPOCH}")
print(f"Eval samples per epoch: {BASELINE_EVAL_SAMPLES}")


# =========================================================
# TRAIN LOOP
# =========================================================
exp_avg = None
best_eval = float("inf")
best_epoch = -1

for epoch in range(1, EPOCHS + 1):
    model.train()
    baseline_model.eval()

    vehicle_penalty = get_vehicle_penalty(epoch)
    start_time = time.time()

    epoch_loss = 0.0
    epoch_train_cost = 0.0
    epoch_baseline_cost = 0.0
    epoch_train_routes = 0.0
    epoch_train_distance = 0.0

    for batch_id in range(1, BATCHES_PER_EPOCH + 1):
        n = sample_nodes()
        locs, demands = make_batch(n, BATCH_SIZE, vehicle_penalty)

        # current model: sampling
        sample_costs, log_probs, sample_routes, sample_distance = run_model(
            model, locs, demands, n, "sampling", vehicle_penalty
        )

        # baseline
        if epoch == 1:
            batch_mean = sample_costs.mean().detach()
            if exp_avg is None:
                exp_avg = batch_mean
            else:
                exp_avg = EXP_BETA * exp_avg + (1 - EXP_BETA) * batch_mean

            baseline_cost = exp_avg.expand_as(sample_costs)
        else:
            with torch.no_grad():
                baseline_cost, _, _, _ = run_model(
                    baseline_model, locs, demands, n, "greedy", vehicle_penalty
                )

        advantage = sample_costs - baseline_cost
        loss = (advantage.detach() * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_train_cost += sample_costs.mean().item()
        epoch_baseline_cost += baseline_cost.mean().item()
        epoch_train_routes += sample_routes.mean().item()
        epoch_train_distance += sample_distance.mean().item()

        if batch_id % 50 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Penalty={vehicle_penalty:.0f} | "
                f"Batch {batch_id:04d}/{BATCHES_PER_EPOCH} | "
                f"nodes={n} | "
                f"avg_cost={epoch_train_cost / batch_id:.4f} | "
                f"avg_base={epoch_baseline_cost / batch_id:.4f} | "
                f"avg_routes={epoch_train_routes / batch_id:.4f} | "
                f"avg_dist={epoch_train_distance / batch_id:.4f} | "
                f"avg_loss={epoch_loss / batch_id:.4f}"
            )

    # -------------------------------
    # END OF EPOCH: baseline update
    # -------------------------------
    model.eval()

    cand_mean, base_mean, p_value = evaluate(model, baseline_model, vehicle_penalty)

    baseline_updated = False
    if cand_mean < base_mean and p_value < BASELINE_ALPHA:
        baseline_model = copy.deepcopy(model).eval()
        for p in baseline_model.parameters():
            p.requires_grad = False
        baseline_updated = True

    epoch_time = time.time() - start_time

    avg_epoch_loss = epoch_loss / BATCHES_PER_EPOCH
    avg_epoch_train_cost = epoch_train_cost / BATCHES_PER_EPOCH
    avg_epoch_baseline_cost = epoch_baseline_cost / BATCHES_PER_EPOCH
    avg_epoch_routes = epoch_train_routes / BATCHES_PER_EPOCH
    avg_epoch_distance = epoch_train_distance / BATCHES_PER_EPOCH

    print(
        f"\nEpoch {epoch:02d} finished | "
        f"Penalty: {vehicle_penalty:.0f} | "
        f"Train Cost: {avg_epoch_train_cost:.4f} | "
        f"Baseline Cost: {avg_epoch_baseline_cost:.4f} | "
        f"Routes: {avg_epoch_routes:.4f} | "
        f"Distance: {avg_epoch_distance:.4f} | "
        f"Loss: {avg_epoch_loss:.4f} | "
        f"Eval Cand: {cand_mean:.4f} | "
        f"Eval Base: {base_mean:.4f} | "
        f"p-value: {p_value:.6f} | "
        f"baseline_updated={baseline_updated} | "
        f"Time: {epoch_time:.2f}s\n"
    )

    # save epoch checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pt")
    save_checkpoint(
        path=ckpt_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        baseline_model=baseline_model,
        stats={
            "vehicle_penalty": vehicle_penalty,
            "avg_loss": avg_epoch_loss,
            "avg_train_cost": avg_epoch_train_cost,
            "avg_baseline_cost": avg_epoch_baseline_cost,
            "avg_routes": avg_epoch_routes,
            "avg_distance": avg_epoch_distance,
            "eval_candidate_cost": cand_mean,
            "eval_baseline_cost": base_mean,
            "p_value": p_value,
            "baseline_updated": baseline_updated,
        },
    )

    # save best by eval
    if cand_mean < best_eval:
        best_eval = cand_mean
        best_epoch = epoch
        best_path = os.path.join(SAVE_DIR, "best_model.pt")
        save_checkpoint(
            path=best_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            baseline_model=baseline_model,
            stats={
                "vehicle_penalty": vehicle_penalty,
                "best_eval_candidate_cost": best_eval,
                "best_epoch": best_epoch,
                "avg_loss": avg_epoch_loss,
                "avg_train_cost": avg_epoch_train_cost,
                "avg_baseline_cost": avg_epoch_baseline_cost,
                "avg_routes": avg_epoch_routes,
                "avg_distance": avg_epoch_distance,
                "eval_candidate_cost": cand_mean,
                "eval_baseline_cost": base_mean,
                "p_value": p_value,
                "baseline_updated": baseline_updated,
            },
        )
        print(f"[INFO] New best model saved at epoch {epoch} with Eval Cand = {cand_mean:.4f}")

final_path = os.path.join(SAVE_DIR, "model_final.pt")
save_checkpoint(
    path=final_path,
    epoch=EPOCHS,
    model=model,
    optimizer=optimizer,
    baseline_model=baseline_model,
    stats={
        "message": "final checkpoint",
        "best_eval_candidate_cost": best_eval,
        "best_epoch": best_epoch,
    },
)

print(f"Training finished. Final model saved to: {final_path}")
print(f"Best checkpoint: epoch {best_epoch}, Eval Cand = {best_eval:.4f}")