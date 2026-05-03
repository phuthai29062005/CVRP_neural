import os
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from read_data import read_data
from caculate import get_good_routes, get_fitness
from Train_Neural.cvrp_model import CVRPModel
from Train_Neural.cvrp_env import CVRPenv


# =========================================================
# INSTANCE CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATH = os.path.join(
    BASE_DIR,
    "ML4VRP2026",
    "Instances",
    "cvrp",
    "vrp",
    "X-n129-k18.vrp",
)

dimension, capacity, nodes = read_data(FILE_PATH)
print(f"[GA] Loaded instance: dimension={dimension}, capacity={capacity}")


# =========================================================
# NEURAL MODEL CACHE
# =========================================================

_NEURAL_MODEL = None
_NEURAL_CKPT_LOADED = None
_NEURAL_LOAD_FAILED = False
_NEURAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_ckpt_path(ckpt_path: Optional[str]) -> str:
    """
    Resolve checkpoint path.
    - absolute path: giữ nguyên
    - relative path: nối từ thư mục chứa GA.py
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(
            "Train_Neural",
            "checkpoints_neural_fill_v2",
            "model_best_sampling.pt",
        )

    if os.path.isabs(ckpt_path):
        return ckpt_path

    return os.path.join(BASE_DIR, ckpt_path)


def _load_neural_model(
    ckpt_path: Optional[str] = None,
    embedding_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 3,
):
    """
    Load neural model một lần, cache lại để GA không phải load checkpoint nhiều lần.
    """
    global _NEURAL_MODEL, _NEURAL_CKPT_LOADED, _NEURAL_LOAD_FAILED

    resolved_ckpt_path = _resolve_ckpt_path(ckpt_path)

    if _NEURAL_MODEL is not None and _NEURAL_CKPT_LOADED == resolved_ckpt_path:
        return _NEURAL_MODEL

    if _NEURAL_LOAD_FAILED:
        raise FileNotFoundError("Neural model was already attempted and failed to load.")

    if not os.path.exists(resolved_ckpt_path):
        _NEURAL_LOAD_FAILED = True
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt_path}")

    model = CVRPModel(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(_NEURAL_DEVICE)

    ckpt = torch.load(resolved_ckpt_path, map_location=_NEURAL_DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    _NEURAL_MODEL = model
    _NEURAL_CKPT_LOADED = resolved_ckpt_path

    print(f"[INFO] Loaded neural model from: {resolved_ckpt_path}")
    return _NEURAL_MODEL


def _make_env(num_nodes: int, capacity_value: float, device: torch.device):
    """
    Tạo CVRPenv. Có fallback để tránh lỗi nếu cvrp_env.py của bạn
    không còn tham số vehicle_penalty/use_vehicle_penalty.
    """
    try:
        return CVRPenv(
            num_nodes=num_nodes,
            capacity=capacity_value,
            device=device,
            use_vehicle_penalty=False,
            vehicle_penalty=0.0,
        )
    except TypeError:
        return CVRPenv(
            num_nodes=num_nodes,
            capacity=capacity_value,
            device=device,
        )


# =========================================================
# BASIC ROUTE UTILS
# =========================================================

def get_vertices_from_routes(routes: Sequence[Sequence[int]]) -> Set[int]:
    vertices = set()

    for route_seg in routes:
        vertices.update(route_seg)

    return vertices


def _euclidean_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    return float(np.hypot(dx, dy))


def _route_distance(route_seg: Sequence[int]) -> float:
    """
    Tính distance của một route: depot -> customers -> depot.
    Depot là node 1.
    """
    if len(route_seg) == 0:
        return float("inf")

    total = 0.0
    prev = 1

    for v in route_seg:
        total += _euclidean_distance(nodes[prev], nodes[v])
        prev = v

    total += _euclidean_distance(nodes[prev], nodes[1])
    return total


def _route_demand(route_seg: Sequence[int]) -> float:
    return float(sum(nodes[v]["demand"] for v in route_seg))


def _fallback_route_score(route_seg: Sequence[int]) -> float:
    """
    Score càng nhỏ càng tốt.

    Dùng distance/customer để tránh việc route ngắn chỉ có 1 customer
    luôn được ưu tiên quá mạnh. Thêm penalty nhẹ nếu route dùng tải kém.
    """
    if len(route_seg) == 0:
        return float("inf")

    dist = _route_distance(route_seg)
    demand = _route_demand(route_seg)

    distance_per_customer = dist / max(len(route_seg), 1)

    load_ratio = demand / max(float(capacity), 1e-9)
    unused_capacity_penalty = max(0.0, 1.0 - load_ratio)

    over_capacity_penalty = 0.0
    if demand > capacity:
        over_capacity_penalty = 1000.0 * ((demand - capacity) / capacity)

    return distance_per_customer + 0.2 * unused_capacity_penalty + over_capacity_penalty


def _safe_score_from_get_good_routes(
    returned_scores: Any,
    route_rank: int,
    route_seg: Sequence[int],
) -> float:
    """
    Nếu get_good_routes trả về score thì dùng score đó.
    Nếu không đọc được score thì tự tính bằng _fallback_route_score().
    """
    if isinstance(returned_scores, (list, tuple)) and route_rank < len(returned_scores):
        candidate_score = returned_scores[route_rank]

        if isinstance(candidate_score, (int, float, np.integer, np.floating)):
            return float(candidate_score)

        if isinstance(candidate_score, (list, tuple)):
            for x in candidate_score:
                if isinstance(x, (int, float, np.integer, np.floating)):
                    return float(x)

    return _fallback_route_score(route_seg)


def _flatten_kept_routes(kept_routes: Sequence[Sequence[int]]) -> Tuple[List[int], List[int]]:
    """
    kept_routes = [[...], [...], ...]
    -> child_permutation + route_markers

    marker = 1 nghĩa là bắt đầu route mới.
    marker = 0 nghĩa là tiếp tục route hiện tại.
    """
    child_permutation = []
    child_route_markers = []

    for route_seg in kept_routes:
        for j, vertex in enumerate(route_seg):
            child_permutation.append(vertex)
            child_route_markers.append(1 if j == 0 else 0)

    return child_permutation, child_route_markers


# =========================================================
# WEIGHTED-GREEDY ROUTE SELECTION
# =========================================================

def _collect_route_candidates(parent, route, parent_indices, num_good_routes: int):
    """
    Lấy good routes từ các parent và gán score cho từng route.

    Mỗi candidate có dạng:
        {
            "route": [...],
            "score": float,
            "parent_idx": int,
            "rank": int,
        }
    """
    candidates = []

    for p_idx in parent_indices:
        good_routes, returned_scores = get_good_routes(
            parent[p_idx],
            route[p_idx],
            nodes,
            num_good_routes=num_good_routes,
        )

        for rank, route_seg in enumerate(good_routes):
            route_seg = list(route_seg)

            if len(route_seg) == 0:
                continue

            route_vertices = set(route_seg)
            if len(route_vertices) != len(route_seg):
                continue

            score = _safe_score_from_get_good_routes(
                returned_scores=returned_scores,
                route_rank=rank,
                route_seg=route_seg,
            )

            candidates.append(
                {
                    "route": route_seg,
                    "score": float(score),
                    "parent_idx": p_idx,
                    "rank": rank,
                }
            )

    candidates.sort(key=lambda x: x["score"])
    return candidates


def _weighted_choice_by_score(candidates: Sequence[Dict[str, Any]], temperature: float):
    """
    Chọn một route theo xác suất.
    Score càng nhỏ thì xác suất càng cao.

    temperature nhỏ -> greedy hơn.
    temperature lớn -> random hơn.
    """
    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return candidates[0]

    scores = np.array([c["score"] for c in candidates], dtype=np.float64)

    score_min = scores.min()
    score_std = scores.std()

    normalized = (scores - score_min) / (score_std + 1e-9)

    temperature = max(float(temperature), 1e-6)
    weights = np.exp(-normalized / temperature)

    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weight_sum

    chosen_idx = np.random.choice(len(candidates), p=weights)
    return candidates[int(chosen_idx)]


def select_good_routes_weighted_greedy(
    route_candidates: Sequence[Dict[str, Any]],
    max_kept_routes: int = 5,
    temperature: float = 0.7,
) -> List[List[int]]:
    """
    Chọn route giữ lại theo kiểu weighted-greedy.

    Ý tưởng:
    - Route score càng tốt thì xác suất được chọn càng cao.
    - Khi chọn một route, tất cả route còn lại có trùng customer với route đó bị loại.
    - Như vậy các kept routes không bao giờ bị trùng customer.
    """
    available = list(route_candidates)
    kept_routes = []
    used_vertices = set()

    while available and len(kept_routes) < max_kept_routes:
        feasible = []

        for cand in available:
            route_vertices = set(cand["route"])

            if route_vertices.isdisjoint(used_vertices):
                feasible.append(cand)

        if not feasible:
            break

        chosen = _weighted_choice_by_score(feasible, temperature=temperature)

        if chosen is None:
            break

        chosen_route = list(chosen["route"])
        chosen_vertices = set(chosen_route)

        kept_routes.append(chosen_route)
        used_vertices.update(chosen_vertices)

        # Loại bỏ route đã chọn và mọi route có trùng customer với used_vertices.
        available = [
            cand
            for cand in available
            if set(cand["route"]).isdisjoint(used_vertices)
        ]

    return kept_routes


# =========================================================
# FILL REMAINING CUSTOMERS
# =========================================================

def _normalize_subproblem_coords(depot_xy: np.ndarray, customer_xy: np.ndarray) -> np.ndarray:
    """
    Normalize tọa độ về [0,1] theo bounding box của subproblem.
    """
    all_xy = np.vstack([depot_xy[None, :], customer_xy]).astype(np.float32)

    min_xy = all_xy.min(axis=0, keepdims=True)
    max_xy = all_xy.max(axis=0, keepdims=True)
    scale = np.maximum(max_xy - min_xy, 1e-8)

    return (all_xy - min_xy) / scale


def _random_fill_remaining(
    remaining_vertices: Sequence[int],
    nodes_data,
    capacity_value: float,
) -> Tuple[List[int], List[int]]:
    """
    Fallback random/greedy fill có capacity check.
    """
    permutation = []
    markers = []

    current_load = 0.0

    for idx, vertex in enumerate(remaining_vertices):
        demand = float(nodes_data[vertex]["demand"])

        if idx == 0:
            permutation.append(vertex)
            markers.append(1)
            current_load = demand
        elif current_load + demand <= capacity_value:
            permutation.append(vertex)
            markers.append(0)
            current_load += demand
        else:
            permutation.append(vertex)
            markers.append(1)
            current_load = demand

    return permutation, markers


@torch.no_grad()
def solve_remaining_with_neural(
    remaining_vertices: Sequence[int],
    nodes_data,
    capacity_value: float,
    ckpt_path: Optional[str] = None,
    decode_type: str = "sampling",
) -> Tuple[List[int], List[int]]:
    """
    Dùng neural để giải residual CVRP trên remaining_vertices.

    Returns:
        neural_permutation
        neural_route_markers
    """
    if len(remaining_vertices) == 0:
        return [], []

    remaining_vertices = sorted(remaining_vertices)

    model = _load_neural_model(ckpt_path=ckpt_path)

    depot_xy = np.array(
        [nodes_data[1]["x"], nodes_data[1]["y"]],
        dtype=np.float32,
    )

    customer_xy = np.array(
        [[nodes_data[v]["x"], nodes_data[v]["y"]] for v in remaining_vertices],
        dtype=np.float32,
    )

    all_xy = _normalize_subproblem_coords(depot_xy, customer_xy)

    locs = torch.tensor(
        all_xy,
        dtype=torch.float32,
        device=_NEURAL_DEVICE,
    ).unsqueeze(0)

    demands = torch.tensor(
        [[nodes_data[v]["demand"] / capacity_value for v in remaining_vertices]],
        dtype=torch.float32,
        device=_NEURAL_DEVICE,
    )

    env = _make_env(
        num_nodes=len(remaining_vertices),
        capacity_value=capacity_value,
        device=_NEURAL_DEVICE,
    )

    env.reset(batch_size=1, locs=locs, demands=demands)

    state = env.get_state()
    embeddings = model._get_embeddings(state["locs"], state["demands"])

    done = torch.zeros(1, dtype=torch.bool, device=_NEURAL_DEVICE)
    action_sequence = []

    while not done.all():
        mask = env.get_mask()

        probs, _ = model.decoder(
            embeddings=embeddings,
            current_node=state["current_node"],
            remaining_capacity=state["remaining_capacity"],
            mask=mask,
        )

        action, _ = model.decoder.select_node(
            probs=probs,
            mask=mask,
            decode_type=decode_type,
        )

        action_sequence.append(action.item())

        state, _, done = env.step(action)

    neural_permutation = []
    neural_route_markers = []

    start_new_route = True

    for action in action_sequence:
        if action == 0:
            start_new_route = True
            continue

        global_vertex = remaining_vertices[action - 1]

        neural_permutation.append(global_vertex)
        neural_route_markers.append(1 if start_new_route else 0)

        start_new_route = False

    return neural_permutation, neural_route_markers


# =========================================================
# MAIN GA CROSSOVER
# =========================================================

def GA(
    parent,
    route,
    par1,
    par2,
    par3,
    use_neural_fill: bool = True,
    neural_ckpt_path: Optional[str] = os.path.join(
        "Train_Neural",
        "checkpoints_neural_fill_v2",
        "model_best_sampling.pt",
    ),
    neural_decode_type: str = "sampling",
    num_good_routes_per_parent: int = 7,
    max_kept_routes: int = 7,
    selection_trials: int = 10,
    route_select_temperature: float = 0.8,
):
    """
    GA crossover:
    1. Lấy good routes từ 3 parent.
    2. Chọn route giữ lại bằng weighted-greedy, không chọn route trùng customer.
    3. Remaining customers được fill bằng neural hoặc random.
    4. Thử nhiều lần và trả child tốt nhất.
    """
    resolved_ckpt_path = _resolve_ckpt_path(neural_ckpt_path)

    route_candidates = _collect_route_candidates(
        parent=parent,
        route=route,
        parent_indices=[par1, par2, par3],
        num_good_routes=num_good_routes_per_parent,
    )

    best_fitness = float("inf")
    best_child = None
    best_child_route = None

    warned_neural_failure = False

    all_vertices = set(range(2, dimension + 1))

    for _ in range(selection_trials):
        kept_routes = select_good_routes_weighted_greedy(
            route_candidates=route_candidates,
            max_kept_routes=max_kept_routes,
            temperature=route_select_temperature,
        )

        used_vertices = get_vertices_from_routes(kept_routes)
        remaining_vertices = sorted(all_vertices - used_vertices)

        child_permutation, child_route_markers = _flatten_kept_routes(kept_routes)

        if remaining_vertices:
            if use_neural_fill:
                try:
                    fill_perm, fill_markers = solve_remaining_with_neural(
                        remaining_vertices=remaining_vertices,
                        nodes_data=nodes,
                        capacity_value=capacity,
                        ckpt_path=resolved_ckpt_path,
                        decode_type=neural_decode_type,
                    )
                except Exception as e:
                    if not warned_neural_failure:
                        print(f"[WARN] Neural fill failed. Fallback to random fill. Error: {e}")
                        warned_neural_failure = True

                    fallback_vertices = list(remaining_vertices)
                    random.shuffle(fallback_vertices)

                    fill_perm, fill_markers = _random_fill_remaining(
                        fallback_vertices,
                        nodes,
                        capacity,
                    )
            else:
                fallback_vertices = list(remaining_vertices)
                random.shuffle(fallback_vertices)

                fill_perm, fill_markers = _random_fill_remaining(
                    fallback_vertices,
                    nodes,
                    capacity,
                )

            child_permutation.extend(fill_perm)
            child_route_markers.extend(fill_markers)

        expected_len = dimension - 1

        if len(child_permutation) != expected_len:
            raise ValueError(
                f"Child permutation length mismatch: "
                f"got {len(child_permutation)}, expected {expected_len}"
            )

        if len(child_route_markers) != expected_len:
            raise ValueError(
                f"Child route marker length mismatch: "
                f"got {len(child_route_markers)}, expected {expected_len}"
            )

        if len(set(child_permutation)) != expected_len:
            raise ValueError("Child has duplicated customers.")

        child_fitness = get_fitness(
            child_permutation,
            child_route_markers,
            nodes,
        )

        if child_fitness < best_fitness:
            best_fitness = child_fitness
            best_child = child_permutation
            best_child_route = child_route_markers

    return best_child, best_child_route, best_fitness