import os
import time
import random
import torch
import torch.optim as optim

from cvrp_model import CVRPModel
from cvrp_env import CVRPenv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# HYPERPARAM
# =====================
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-4
BATCHES_PER_EPOCH = 1000

# Train cho distribution remaining nodes của GA
TRAIN_NODE_SIZES = [100, 120, 150]

SAVE_DIR = "checkpoints_neural_fill"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# MODEL
# =====================
model = CVRPModel(
    embedding_dim=128,
    num_heads=8,
    num_layers=3
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Device: {device}")
print(f"Train node sizes: {TRAIN_NODE_SIZES}")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Batches/epoch: {BATCHES_PER_EPOCH}")


# =====================
# TRAIN LOOP
# =====================
for epoch in range(EPOCHS):
    model.train()
    start_time = time.time()

    epoch_loss = 0.0
    epoch_reward = 0.0
    epoch_cost = 0.0

    for batch_id in range(BATCHES_PER_EPOCH):
        num_nodes = random.choice(TRAIN_NODE_SIZES)

        # capacity cho env:
        # code env của bạn có chuẩn mặc định cho 20/50/100,
        # còn >100 sẽ fallback về 50.0 nếu không truyền tay.
        # Với case neural fill 100–150, để đơn giản có thể giữ 50.0.
        # Nếu sau này muốn chỉnh sát hơn với GA residual subproblem thì đổi chỗ này.
        env = CVRPenv(num_nodes=num_nodes, capacity=50.0, device=device)
        env.reset(batch_size=BATCH_SIZE)

        # model forward chuẩn của bạn trả:
        # rewards: [B] (reward âm = -tour length)
        # log_probs: [B]
        rewards, log_probs = model(env, decode_type="sampling")

        # REINFORCE baseline đơn giản theo mean của batch
        baseline = rewards.mean()
        advantage = rewards - baseline

        # reward âm: reward càng lớn (ít âm hơn) càng tốt
        # maximize E[reward] <=> minimize negative objective
        loss = -(advantage.detach() * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_reward = rewards.mean().item()
        batch_cost = (-rewards).mean().item()

        epoch_loss += loss.item()
        epoch_reward += batch_reward
        epoch_cost += batch_cost

        if (batch_id + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1:02d}/{EPOCHS} | "
                f"Batch {batch_id+1:04d}/{BATCHES_PER_EPOCH} | "
                f"nodes={num_nodes} | "
                f"avg_reward={epoch_reward/(batch_id+1):.4f} | "
                f"avg_cost={epoch_cost/(batch_id+1):.4f} | "
                f"avg_loss={epoch_loss/(batch_id+1):.4f}"
            )

    epoch_time = time.time() - start_time
    avg_epoch_loss = epoch_loss / BATCHES_PER_EPOCH
    avg_epoch_reward = epoch_reward / BATCHES_PER_EPOCH
    avg_epoch_cost = epoch_cost / BATCHES_PER_EPOCH

    print(
        f"\nEpoch {epoch+1:02d} finished | "
        f"Avg Reward: {avg_epoch_reward:.4f} | "
        f"Avg Cost: {avg_epoch_cost:.4f} | "
        f"Avg Loss: {avg_epoch_loss:.4f} | "
        f"Time: {epoch_time:.2f}s\n"
    )

    # save mỗi epoch
    ckpt_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_epoch_loss,
            "avg_cost": avg_epoch_cost,
            "train_node_sizes": TRAIN_NODE_SIZES,
        },
        ckpt_path,
    )

# save final
final_path = os.path.join(SAVE_DIR, "model_final.pt")
torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_node_sizes": TRAIN_NODE_SIZES,
    },
    final_path,
)

print(f"Training finished. Final model saved to: {final_path}")