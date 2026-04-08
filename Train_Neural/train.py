import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.optim as optim

try:
    from scipy.stats import ttest_rel
except Exception:
    ttest_rel = None

from cvrp_env import CVRPenv
from cvrp_model import CVRPModel


@dataclass
class TrainConfig:
    num_nodes: int = 50            # 50 or 100 for now
    capacity: Optional[float] = None

    embedding_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3

    lr: float = 1e-4
    epochs: int = 100
    batches_per_epoch: int = 2500

    batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    eval_size: int = 10_000

    warmup_beta: float = 0.8
    significance_level: float = 0.05
    grad_clip_norm: float = 1.0

    save_dir: str = "checkpoints"
    save_name: Optional[str] = None
    print_every: int = 100


def get_default_capacity(num_nodes: int) -> float:
    if num_nodes == 50:
        return 40.0
    if num_nodes == 100:
        return 50.0
    raise ValueError("This training file is prepared for 50 or 100 nodes only.")


def get_default_batch_size(num_nodes: int) -> int:
    if num_nodes == 50:
        return 512
    if num_nodes == 100:
        return 256
    raise ValueError("This training file is prepared for 50 or 100 nodes only.")


def make_env(num_nodes: int, capacity: float, device: torch.device) -> CVRPenv:
    return CVRPenv(num_nodes=num_nodes, capacity=capacity, device=device)


def make_eval_dataset(num_nodes: int, capacity: float, eval_size: int, device: torch.device):
    env = make_env(num_nodes, capacity, device)
    env.reset(batch_size=eval_size)
    return env.locs.clone(), env.demands.clone()


def rollout_on_batch(
    model: CVRPModel,
    num_nodes: int,
    capacity: float,
    batch_size: int,
    device: torch.device,
    decode_type: str,
    locs: Optional[torch.Tensor] = None,
    demands: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    env = make_env(num_nodes, capacity, device)
    if locs is None or demands is None:
        env.reset(batch_size=batch_size)
    else:
        env.reset(batch_size=batch_size, locs=locs, demands=demands)

    rewards, log_probs = model(env, decode_type=decode_type)
    costs = -rewards  # reward = -tour_length
    return costs, log_probs


@torch.no_grad()
def evaluate_greedy(
    model: CVRPModel,
    num_nodes: int,
    capacity: float,
    eval_locs: torch.Tensor,
    eval_demands: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs = []
    total = eval_locs.size(0)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_locs = eval_locs[start:end]
        batch_demands = eval_demands[start:end]
        costs, _ = rollout_on_batch(
            model=model,
            num_nodes=num_nodes,
            capacity=capacity,
            batch_size=end - start,
            device=device,
            decode_type="greedy",
            locs=batch_locs,
            demands=batch_demands,
        )
        outputs.append(costs)

    return torch.cat(outputs, dim=0)


def one_sided_paired_t_test(current_costs: torch.Tensor, baseline_costs: torch.Tensor):
    """
    H0: current >= baseline
    H1: current < baseline
    """
    diff_mean = float((current_costs - baseline_costs).mean().item())

    if ttest_rel is None:
        # Fallback: no scipy installed -> only use mean improvement, no significance guarantee.
        improved = diff_mean < 0.0
        return improved, 1.0

    res = ttest_rel(current_costs.cpu().numpy(), baseline_costs.cpu().numpy(), nan_policy="raise")
    p_two_sided = float(res.pvalue)

    if diff_mean < 0.0:
        p_one_sided = p_two_sided / 2.0
    else:
        p_one_sided = 1.0 - p_two_sided / 2.0

    improved = diff_mean < 0.0
    return improved, p_one_sided


def save_checkpoint(model: CVRPModel, cfg: TrainConfig, epoch: int, eval_cost: float) -> str:
    os.makedirs(cfg.save_dir, exist_ok=True)

    filename = cfg.save_name or f"am_cvrp_{cfg.num_nodes}.pt"
    path = os.path.join(cfg.save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "eval_cost": eval_cost,
        },
        path,
    )
    return path


def train_cvrp_paper_style(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    capacity = float(cfg.capacity if cfg.capacity is not None else get_default_capacity(cfg.num_nodes))
    batch_size = int(cfg.batch_size if cfg.batch_size is not None else get_default_batch_size(cfg.num_nodes))
    eval_batch_size = int(cfg.eval_batch_size if cfg.eval_batch_size is not None else batch_size)

    print("=" * 80)
    print("Training Attention Model for CVRP")
    print(f"device            : {device}")
    print(f"num_nodes         : {cfg.num_nodes}")
    print(f"capacity          : {capacity}")
    print(f"batch_size        : {batch_size}")
    print(f"eval_batch_size   : {eval_batch_size}")
    print(f"epochs            : {cfg.epochs}")
    print(f"batches_per_epoch : {cfg.batches_per_epoch}")
    print(f"eval_size         : {cfg.eval_size}")
    print("=" * 80)

    train_model = CVRPModel(
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
    ).to(device)

    baseline_model = copy.deepcopy(train_model).to(device)
    baseline_model.eval()

    optimizer = optim.Adam(train_model.parameters(), lr=cfg.lr)

    eval_locs, eval_demands = make_eval_dataset(
        num_nodes=cfg.num_nodes,
        capacity=capacity,
        eval_size=cfg.eval_size,
        device=device,
    )

    moving_avg_baseline = None
    best_eval_cost = math.inf

    for epoch in range(cfg.epochs):
        train_model.train()
        baseline_model.eval()

        epoch_loss = 0.0
        epoch_cost = 0.0

        for batch_idx in range(cfg.batches_per_epoch):
            # Generate a fresh batch on the fly
            env = make_env(cfg.num_nodes, capacity, device)
            env.reset(batch_size=batch_size)
            locs = env.locs.clone()
            demands = env.demands.clone()

            # Train rollout: sampling
            train_costs, log_probs = rollout_on_batch(
                model=train_model,
                num_nodes=cfg.num_nodes,
                capacity=capacity,
                batch_size=batch_size,
                device=device,
                decode_type="sampling",
                locs=locs,
                demands=demands,
            )

            # Baseline
            if epoch == 0:
                batch_mean_cost = train_costs.mean().detach()
                if moving_avg_baseline is None:
                    moving_avg_baseline = batch_mean_cost
                else:
                    moving_avg_baseline = cfg.warmup_beta * moving_avg_baseline + (1.0 - cfg.warmup_beta) * batch_mean_cost
                baseline = moving_avg_baseline
            else:
                with torch.no_grad():
                    baseline_costs, _ = rollout_on_batch(
                        model=baseline_model,
                        num_nodes=cfg.num_nodes,
                        capacity=capacity,
                        batch_size=batch_size,
                        device=device,
                        decode_type="greedy",
                        locs=locs,
                        demands=demands,
                    )
                baseline = baseline_costs

            advantage = (train_costs - baseline).detach()
            loss = (advantage * log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_cost += float(train_costs.mean().item())

            if (batch_idx + 1) % cfg.print_every == 0:
                print(
                    f"[Epoch {epoch+1:03d}/{cfg.epochs:03d}] "
                    f"Batch {batch_idx+1:04d}/{cfg.batches_per_epoch:04d} | "
                    f"AvgCostSoFar: {epoch_cost/(batch_idx+1):.4f} | "
                    f"AvgLossSoFar: {epoch_loss/(batch_idx+1):.4f}"
                )

        avg_epoch_loss = epoch_loss / cfg.batches_per_epoch
        avg_epoch_cost = epoch_cost / cfg.batches_per_epoch

        print(
            f"\nEpoch {epoch+1:03d} finished | "
            f"Train Avg Cost: {avg_epoch_cost:.4f} | "
            f"Train Avg Loss: {avg_epoch_loss:.4f}"
        )

        # Evaluate current model on fixed eval set
        train_model.eval()
        current_eval_costs = evaluate_greedy(
            model=train_model,
            num_nodes=cfg.num_nodes,
            capacity=capacity,
            eval_locs=eval_locs,
            eval_demands=eval_demands,
            batch_size=eval_batch_size,
            device=device,
        )
        current_eval_mean = float(current_eval_costs.mean().item())

        if epoch == 0:
            baseline_model.load_state_dict(train_model.state_dict())
            baseline_model.eval()
            best_eval_cost = current_eval_mean
            ckpt_path = save_checkpoint(baseline_model, cfg, epoch + 1, current_eval_mean)
            print(
                f"[Epoch 001] Warmup complete -> initialize rollout baseline | "
                f"Eval Cost: {current_eval_mean:.4f}"
            )
            print(f"Saved checkpoint: {ckpt_path}\n")
            continue

        baseline_eval_costs = evaluate_greedy(
            model=baseline_model,
            num_nodes=cfg.num_nodes,
            capacity=capacity,
            eval_locs=eval_locs,
            eval_demands=eval_demands,
            batch_size=eval_batch_size,
            device=device,
        )
        baseline_eval_mean = float(baseline_eval_costs.mean().item())

        improved, p_value = one_sided_paired_t_test(current_eval_costs, baseline_eval_costs)

        print(
            f"Eval current  : {current_eval_mean:.4f}\n"
            f"Eval baseline : {baseline_eval_mean:.4f}\n"
            f"Improved      : {improved}\n"
            f"p-value       : {p_value:.6f}"
        )

        if improved and p_value < cfg.significance_level:
            baseline_model.load_state_dict(train_model.state_dict())
            baseline_model.eval()

            # Refresh eval set after accepting a new baseline
            eval_locs, eval_demands = make_eval_dataset(
                num_nodes=cfg.num_nodes,
                capacity=capacity,
                eval_size=cfg.eval_size,
                device=device,
            )

            best_eval_cost = current_eval_mean
            ckpt_path = save_checkpoint(baseline_model, cfg, epoch + 1, current_eval_mean)
            print(f"Baseline updated at epoch {epoch+1:03d}.")
            print(f"Saved checkpoint: {ckpt_path}\n")
        else:
            print(f"Baseline kept unchanged at epoch {epoch+1:03d}.\n")

    print(f"Training complete. Best accepted baseline eval cost: {best_eval_cost:.4f}")
    return baseline_model, train_model


if __name__ == "__main__":
    # Train CVRP50 first
    cfg = TrainConfig(
        num_nodes=50,
        capacity=40.0,
        batch_size=512,
        eval_batch_size=512,
        epochs=100,
        batches_per_epoch=2500,
        save_name="am_cvrp50.pt",
    )
    train_cvrp_paper_style(cfg)

    # To train CVRP100 later, comment the block above and use this block instead:
    # cfg = TrainConfig(
    #     num_nodes=100,
    #     capacity=50.0,
    #     batch_size=256,
    #     eval_batch_size=256,
    #     epochs=100,
    #     batches_per_epoch=2500,
    #     save_name="am_cvrp100.pt",
    # )
    # train_cvrp_paper_style(cfg)
