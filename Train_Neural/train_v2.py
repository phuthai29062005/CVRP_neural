import os
import time
import random
import torch
import torch.optim as optim

from cvrp_model import CVRPModel
from cvrp_env import CVRPenv


# =========================================================
# DEVICE CONFIG FOR RTX 3060
# =========================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device("cpu")
    print("[WARNING] CUDA is not available. Training will run on CPU.")
    print("If you are on RTX 3060, check your PyTorch CUDA installation.")


# =========================================================
# SETTINGS
# =========================================================

SEED = 2026
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# RTX 3060 12GB safe setting
BASE_BATCH_SIZE = 64
ROLLOUTS_PER_INSTANCE = 8
EFFECTIVE_BATCH = BASE_BATCH_SIZE * ROLLOUTS_PER_INSTANCE

TRAIN_NODE_SIZES = [50, 80, 100, 120, 150]

CAPACITY = {
    50: 40,
    80: 45,
    100: 50,
    120: 55,
    150: 60,
}

# Full training config for RTX 3060
EPOCHS = 70
BATCHES_PER_EPOCH = 1500

# Nếu muốn test nhanh trước, đổi tạm:
# EPOCHS = 2
# BATCHES_PER_EPOCH = 20

LR = 3e-5
GRAD_CLIP = 1.0
WARMUP_EPOCHS = 2

LOG_EVERY = 100
SAMPLING_VALIDATE_EVERY = 5

VAL_BATCH_SIZE = 128
VAL_NODE_SIZES = [50, 100, 150]
VAL_MODES = ["uniform", "cluster"]

USE_AMP = torch.cuda.is_available()

SAVE_DIR = "checkpoints_neural_fill_v2"
os.makedirs(SAVE_DIR, exist_ok=True)


RESUME = True
RESUME_PATH = os.path.join(SAVE_DIR, "model_latest.pt")

# =========================================================
# DATA GENERATION
# =========================================================

def generate_locs(batch_size, num_nodes, device, mode="uniform"):
    """
    Return:
        locs: [batch_size, num_nodes + 1, 2]
        locs[:, 0, :] = depot
        locs[:, 1:, :] = customers
    """

    depot = torch.rand(batch_size, 1, 2, device=device)

    if mode == "uniform":
        customers = torch.rand(batch_size, num_nodes, 2, device=device)

    elif mode == "cluster":
        num_clusters = 4

        centers = torch.rand(batch_size, num_clusters, 2, device=device)

        assign = torch.randint(
            low=0,
            high=num_clusters,
            size=(batch_size, num_nodes),
            device=device,
        )

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)

        customers = centers[batch_idx, assign]

        customers = customers + 0.07 * torch.randn(
            batch_size, num_nodes, 2, device=device
        )

        customers = customers.clamp(0.0, 1.0)

    elif mode == "centered":
        customers = 0.5 + 0.18 * torch.randn(
            batch_size, num_nodes, 2, device=device
        )

        customers = customers.clamp(0.0, 1.0)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    locs = torch.cat([depot, customers], dim=1)

    return locs


def make_batch(batch_size, num_nodes, device, mode="uniform"):
    """
    Return:
        locs: [B, N+1, 2]
        demands: [B, N]
        capacity: float
    """

    capacity = float(CAPACITY[num_nodes])

    locs = generate_locs(
        batch_size=batch_size,
        num_nodes=num_nodes,
        device=device,
        mode=mode,
    )

    raw_demands = torch.randint(
        low=1,
        high=10,
        size=(batch_size, num_nodes),
        device=device,
    )

    demands = raw_demands.float() / capacity

    return locs, demands, capacity


def choose_distribution():
    """
    60% uniform, 30% cluster, 10% centered.
    """

    r = random.random()

    if r < 0.60:
        return "uniform"
    elif r < 0.90:
        return "cluster"
    else:
        return "centered"


# =========================================================
# VALIDATION
# =========================================================

def build_validation_sets():
    """
    Fixed validation sets.
    Không random lại mỗi epoch để so sánh công bằng.
    """

    val_sets = []

    # Tạo validation cố định với seed riêng.
    cpu_rng_state = torch.random.get_rng_state()
    py_rng_state = random.getstate()

    torch.manual_seed(999)
    random.seed(999)

    for n in VAL_NODE_SIZES:
        for mode in VAL_MODES:
            locs, demands, capacity = make_batch(
                batch_size=VAL_BATCH_SIZE,
                num_nodes=n,
                device=device,
                mode=mode,
            )

            val_sets.append(
                {
                    "num_nodes": n,
                    "mode": mode,
                    "capacity": capacity,
                    "locs": locs.detach(),
                    "demands": demands.detach(),
                }
            )

    torch.random.set_rng_state(cpu_rng_state)
    random.setstate(py_rng_state)

    return val_sets


@torch.no_grad()
def validate_greedy(model, val_sets):
    """
    Greedy validation.
    """

    model.eval()

    total_cost = 0.0
    count = 0
    detail = {}

    for item in val_sets:
        n = item["num_nodes"]
        mode = item["mode"]
        capacity = item["capacity"]
        locs = item["locs"]
        demands = item["demands"]

        env = CVRPenv(
            num_nodes=n,
            capacity=capacity,
            device=device,
        )

        env.reset(
            batch_size=locs.size(0),
            locs=locs,
            demands=demands,
        )

        rewards, _ = model(env, decode_type="greedy")

        cost = (-rewards).mean().item()

        key = f"N{n}_{mode}"
        detail[key] = cost

        total_cost += cost
        count += 1

    model.train()

    avg_cost = total_cost / count

    return avg_cost, detail


@torch.no_grad()
def validate_sampling_best(model, val_sets, samples=64, chunk_samples=8):
    """
    Sampling validation: sample nhiều nghiệm rồi lấy best.
    Chia chunk để tránh nổ VRAM.
    """

    model.eval()

    total_cost = 0.0
    count = 0
    detail = {}

    for item in val_sets:
        n = item["num_nodes"]
        mode = item["mode"]
        capacity = item["capacity"]
        locs = item["locs"]
        demands = item["demands"]

        batch_size = locs.size(0)

        best_cost = torch.full(
            (batch_size,),
            float("inf"),
            device=device,
        )

        remaining = samples

        while remaining > 0:
            cur_samples = min(chunk_samples, remaining)

            locs_rep = locs.repeat_interleave(cur_samples, dim=0)
            demands_rep = demands.repeat_interleave(cur_samples, dim=0)

            env = CVRPenv(
                num_nodes=n,
                capacity=capacity,
                device=device,
            )

            env.reset(
                batch_size=batch_size * cur_samples,
                locs=locs_rep,
                demands=demands_rep,
            )

            rewards, _ = model(env, decode_type="sampling")
            costs = (-rewards).reshape(batch_size, cur_samples)

            best_cost = torch.minimum(best_cost, costs.min(dim=1).values)

            remaining -= cur_samples

        avg_best_cost = best_cost.mean().item()

        key = f"N{n}_{mode}_best{samples}"
        detail[key] = avg_best_cost

        total_cost += avg_best_cost
        count += 1

    model.train()

    avg_cost = total_cost / count

    return avg_cost, detail


# =========================================================
# MODEL + OPTIMIZER
# =========================================================

model = CVRPModel(
    embedding_dim=128,
    num_heads=8,
    num_layers=3,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS-WARMUP_EPOCHS,
    eta_min=1e-5,
)

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

val_sets = build_validation_sets()

best_val_greedy = float("inf")
best_val_sampling = float("inf")

START_EPOCH = 1

if RESUME and os.path.exists(RESUME_PATH):
    print(f"Loading checkpoint from: {RESUME_PATH}")

    checkpoint = torch.load(
        RESUME_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_val_greedy = checkpoint.get("best_val_greedy", best_val_greedy)
    best_val_sampling = checkpoint.get("best_val_sampling", best_val_sampling)

    START_EPOCH = checkpoint["epoch"] + 1

    print(f"Resumed from epoch {checkpoint['epoch']}")
    print(f"Continue training from epoch {START_EPOCH}")
    print(f"Best greedy so far: {best_val_greedy}")
    print(f"Best sampling so far: {best_val_sampling}")

else:
    print("No checkpoint loaded. Training from scratch.")



print("=" * 80)
print("TRAIN CONFIG")
print(f"Device: {device}")
print(f"BASE_BATCH_SIZE: {BASE_BATCH_SIZE}")
print(f"ROLLOUTS_PER_INSTANCE: {ROLLOUTS_PER_INSTANCE}")
print(f"EFFECTIVE_BATCH: {EFFECTIVE_BATCH}")
print(f"TRAIN_NODE_SIZES: {TRAIN_NODE_SIZES}")
print(f"EPOCHS: {EPOCHS}")
print(f"BATCHES_PER_EPOCH: {BATCHES_PER_EPOCH}")
print(f"LR: {LR}")
print(f"USE_AMP: {USE_AMP}")
print("=" * 80)


# =========================================================
# TRAIN LOOP
# =========================================================

for epoch in range(START_EPOCH - 1, EPOCHS):
    
    current_epoch = epoch + 1

    # LR warmup chỉ chạy khi train từ đầu, không chạy lại khi resume
    if current_epoch <= WARMUP_EPOCHS and START_EPOCH == 1:
        warmup_lr = LR * current_epoch / WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group["lr"] = warmup_lr
            
    model.train()

    start_time = time.time()

    epoch_loss = 0.0
    epoch_cost = 0.0

    for batch_id in range(BATCHES_PER_EPOCH):

        # -----------------------------
        # 1. Sample problem setting
        # -----------------------------
        num_nodes = random.choice(TRAIN_NODE_SIZES)
        mode = choose_distribution()

        # -----------------------------
        # 2. Create original batch
        # -----------------------------
        locs, demands, capacity = make_batch(
            batch_size=BASE_BATCH_SIZE,
            num_nodes=num_nodes,
            device=device,
            mode=mode,
        )

        # -----------------------------
        # 3. Repeat each instance K times
        # -----------------------------
        locs_rep = locs.repeat_interleave(ROLLOUTS_PER_INSTANCE, dim=0)
        demands_rep = demands.repeat_interleave(ROLLOUTS_PER_INSTANCE, dim=0)

        # -----------------------------
        # 4. Environment
        # -----------------------------
        env = CVRPenv(
            num_nodes=num_nodes,
            capacity=capacity,
            device=device,
        )

        env.reset(
            batch_size=EFFECTIVE_BATCH,
            locs=locs_rep,
            demands=demands_rep,
        )

        # -----------------------------
        # 5. Forward + loss
        # -----------------------------
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            rewards, log_probs = model(env, decode_type="sampling")

            rewards = rewards.reshape(BASE_BATCH_SIZE, ROLLOUTS_PER_INSTANCE)
            log_probs = log_probs.reshape(BASE_BATCH_SIZE, ROLLOUTS_PER_INSTANCE)

            baseline = rewards.mean(dim=1, keepdim=True)
            advantages = rewards - baseline

            loss = -(advantages.detach() * log_probs).mean()

        # -----------------------------
        # 6. Backprop
        # -----------------------------
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        # -----------------------------
        # 7. Logging
        # -----------------------------
        batch_cost = (-rewards).mean().item()

        epoch_loss += loss.item()
        epoch_cost += batch_cost

        if (batch_id + 1) % LOG_EVERY == 0:
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                f"Batch {batch_id + 1:04d}/{BATCHES_PER_EPOCH} | "
                f"N={num_nodes:<3d} | "
                f"mode={mode:<8s} | "
                f"cap={capacity:<4.0f} | "
                f"avg_cost={epoch_cost / (batch_id + 1):.4f} | "
                f"avg_loss={epoch_loss / (batch_id + 1):.4f}"
            )

    if current_epoch > WARMUP_EPOCHS:
        scheduler.step()

    # =====================================================
    # END OF EPOCH: VALIDATION
    # =====================================================

    epoch_time = time.time() - start_time

    avg_train_cost = epoch_cost / BATCHES_PER_EPOCH
    avg_train_loss = epoch_loss / BATCHES_PER_EPOCH

    val_greedy, val_greedy_detail = validate_greedy(model, val_sets)
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"\nEpoch {epoch + 1:03d}/{EPOCHS} finished | "
        f"train_cost={avg_train_cost:.4f} | "
        f"train_loss={avg_train_loss:.4f} | "
        f"val_greedy={val_greedy:.4f} | "
        f"lr={current_lr:.6f} | "
        f"time={epoch_time:.2f}s"
    )
    print(f"Validation greedy detail: {val_greedy_detail}")

    # =====================================================
    # SAVE BEST GREEDY MODEL
    # =====================================================

    if val_greedy < best_val_greedy:
        best_val_greedy = val_greedy

        best_path = os.path.join(SAVE_DIR, "model_best_greedy.pt")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_greedy": val_greedy,
                "val_greedy_detail": val_greedy_detail,
                "train_node_sizes": TRAIN_NODE_SIZES,
                "base_batch_size": BASE_BATCH_SIZE,
                "rollouts_per_instance": ROLLOUTS_PER_INSTANCE,
                "capacity": CAPACITY,
            },
            best_path,
        )

        print(f"Saved best greedy model: {best_path}")

    # =====================================================
    # SAMPLING VALIDATION EVERY FEW EPOCHS
    # =====================================================


    if (epoch + 1) % SAMPLING_VALIDATE_EVERY == 0:
        val_sampling, val_sampling_detail = validate_sampling_best(
            model,
            val_sets,
            samples=64,
            chunk_samples=8,
        )

        print(f"Validation sampling best-of-64: {val_sampling:.4f}")
        print(f"Validation sampling detail: {val_sampling_detail}")

        if val_sampling < best_val_sampling:
            best_val_sampling = val_sampling

            best_sampling_path = os.path.join(SAVE_DIR, "model_best_sampling.pt")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_sampling_best64": val_sampling,
                    "val_sampling_detail": val_sampling_detail,
                    "train_node_sizes": TRAIN_NODE_SIZES,
                    "base_batch_size": BASE_BATCH_SIZE,
                    "rollouts_per_instance": ROLLOUTS_PER_INSTANCE,
                    "capacity": CAPACITY,
                },
                best_sampling_path,
            )

            print(f"Saved best sampling model: {best_sampling_path}")

    # =====================================================
    # SAVE LATEST + EPOCH CHECKPOINT
    # =====================================================

    latest_path = os.path.join(SAVE_DIR, "model_latest.pt")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "avg_train_cost": avg_train_cost,
            "avg_train_loss": avg_train_loss,
            "val_greedy": val_greedy,
            "val_greedy_detail": val_greedy_detail,
            "best_val_greedy": best_val_greedy,
            "best_val_sampling": best_val_sampling,
            "train_node_sizes": TRAIN_NODE_SIZES,
            "base_batch_size": BASE_BATCH_SIZE,
            "rollouts_per_instance": ROLLOUTS_PER_INSTANCE,
            "capacity": CAPACITY,
        },
        latest_path,
    )


    print(f"Saved latest checkpoint: {latest_path}")


# =========================================================
# SAVE FINAL MODEL
# =========================================================


print("=" * 80)
print("Training finished.")
print(f"Best greedy validation cost: {best_val_greedy:.4f}")
print(f"Best sampling validation cost: {best_val_sampling:.4f}")
print("=" * 80)