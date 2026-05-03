import os
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from Train_Neural.cvrp_model import CVRPModel
from Train_Neural.cvrp_env import CVRPenv


# =========================================================
# CONSTANTS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEPOT_ID = 1
ROUTE_PENALTY = 1000.0


# =========================================================
# NEURAL MODEL CACHE
# =========================================================

_NEURAL_MODEL = None
_NEURAL_CKPT_LOADED = None
_NEURAL_FAILED_CKPTS: Set[str] = set()
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
    global _NEURAL_MODEL, _NEURAL_CKPT_LOADED

    resolved_ckpt_path = _resolve_ckpt_path(ckpt_path)

    if _NEURAL_MODEL is not None and _NEURAL_CKPT_LOADED == resolved_ckpt_path:
        return _NEURAL_MODEL

    if resolved_ckpt_path in _NEURAL_FAILED_CKPTS:
        raise FileNotFoundError(
            f"Neural checkpoint was already attempted and failed: {resolved_ckpt_path}"
        )

    if not os.path.exists(resolved_ckpt_path):
        _NEURAL_FAILED_CKPTS.add(resolved_ckpt_path)
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt_path}")

    model = CVRPModel(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(_NEURAL_DEVICE)

    try:
        ckpt = torch.load(
            resolved_ckpt_path,
            map_location=_NEURAL_DEVICE,
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            resolved_ckpt_path,
            map_location=_NEURAL_DEVICE,
        )

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
    Tạo CVRPenv. Có fallback để tránh lỗi nếu cvrp_env.py
    không có use_vehicle_penalty / vehicle_penalty.
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
# DISTANCE + ROUTE UTILS
# =========================================================

def _distance(
    nodes_data: Dict[int, Dict[str, Any]],
    a: int,
    b: int,
    dist_matrix=None,
) -> float:
    """
    Lấy distance giữa 2 node.
    Ưu tiên dùng dist_matrix đã precompute.
    """
    if dist_matrix is not None:
        return float(dist_matrix[a, b])

    dx = float(nodes_data[a]["x"]) - float(nodes_data[b]["x"])
    dy = float(nodes_data[a]["y"]) - float(nodes_data[b]["y"])

    return float(np.hypot(dx, dy))


def _separate_routes(
    permutation: Sequence[int],
    route_markers: Sequence[int],
) -> List[List[int]]:
    """
    Tách permutation + route markers thành list routes.
    marker = 1 nghĩa là bắt đầu route mới.
    """
    routes = []
    current_route = []

    for customer, marker in zip(permutation, route_markers):
        if marker == 1:
            if current_route:
                routes.append(current_route)

            current_route = [customer]
        else:
            current_route.append(customer)

    if current_route:
        routes.append(current_route)

    return routes


def _route_distance(
    route_seg: Sequence[int],
    nodes_data: Dict[int, Dict[str, Any]],
    dist_matrix=None,
) -> float:
    """
    Tính distance của 1 route:
        depot -> customers -> depot
    Depot là node 1.
    """
    if len(route_seg) == 0:
        return 0.0

    total = 0.0
    prev = DEPOT_ID

    for customer in route_seg:
        total += _distance(nodes_data, prev, customer, dist_matrix)
        prev = customer

    total += _distance(nodes_data, prev, DEPOT_ID, dist_matrix)

    return total


def _route_demand(
    route_seg: Sequence[int],
    nodes_data: Dict[int, Dict[str, Any]],
) -> float:
    return float(sum(nodes_data[v]["demand"] for v in route_seg))


def get_vertices_from_routes(routes: Sequence[Sequence[int]]) -> Set[int]:
    vertices = set()

    for route_seg in routes:
        vertices.update(route_seg)

    return vertices


def _solution_fitness(
    permutation: Sequence[int],
    route_markers: Sequence[int],
    nodes_data: Dict[int, Dict[str, Any]],
    dist_matrix=None,
) -> float:
    """
    Objective:
        fitness = 1000 * number_of_routes + total_distance
    """
    routes = _separate_routes(permutation, route_markers)

    total_distance = 0.0
    for r in routes:
        total_distance += _route_distance(r, nodes_data, dist_matrix)

    return ROUTE_PENALTY * len(routes) + total_distance


def _route_score(
    route_seg: Sequence[int],
    nodes_data: Dict[int, Dict[str, Any]],
    capacity_value: float,
    dist_matrix=None,
) -> float:
    """
    Score càng nhỏ càng tốt.

    Dùng:
        distance/customer
        + penalty nhẹ nếu route dùng tải kém
        + penalty nếu quá capacity
    """
    if len(route_seg) == 0:
        return float("inf")

    dist = _route_distance(route_seg, nodes_data, dist_matrix)
    demand = _route_demand(route_seg, nodes_data)

    distance_per_customer = dist / max(len(route_seg), 1)

    load_ratio = demand / max(float(capacity_value), 1e-9)
    unused_capacity_penalty = max(0.0, 1.0 - load_ratio)

    over_capacity_penalty = 0.0
    if demand > capacity_value:
        over_capacity_penalty = 1000.0 * ((demand - capacity_value) / capacity_value)

    return distance_per_customer + 0.2 * unused_capacity_penalty + over_capacity_penalty


def _flatten_kept_routes(
    kept_routes: Sequence[Sequence[int]],
) -> Tuple[List[int], List[int]]:
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

def _collect_route_candidates(
    parent,
    route,
    parent_indices: Sequence[int],
    nodes_data: Dict[int, Dict[str, Any]],
    capacity_value: float,
    dist_matrix=None,
    num_good_routes: int = 7,
) -> List[Dict[str, Any]]:
    """
    Lấy good routes từ các parent và gán score cho từng route.

    Mỗi candidate có dạng:
        {
            "route": [...],
            "score": float,
            "parent_idx": int,
            "rank": int,
        }

    Không gọi get_good_routes() nữa để tránh tính euclid lặp lại.
    Dùng trực tiếp dist_matrix nếu có.
    """
    candidates = []

    for p_idx in parent_indices:
        routes = _separate_routes(parent[p_idx], route[p_idx])

        scored_routes = []

        for route_seg in routes:
            route_seg = list(route_seg)

            if len(route_seg) == 0:
                continue

            route_vertices = set(route_seg)

            if len(route_vertices) != len(route_seg):
                continue

            score = _route_score(
                route_seg=route_seg,
                nodes_data=nodes_data,
                capacity_value=capacity_value,
                dist_matrix=dist_matrix,
            )

            scored_routes.append((score, route_seg))

        scored_routes.sort(key=lambda x: x[0])

        for rank, (score, route_seg) in enumerate(scored_routes[:num_good_routes]):
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


def _weighted_choice_by_score(
    candidates: Sequence[Dict[str, Any]],
    temperature: float,
):
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
    max_kept_routes: int = 7,
    temperature: float = 0.8,
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

        available = [
            cand
            for cand in available
            if set(cand["route"]).isdisjoint(used_vertices)
        ]

    return kept_routes


# =========================================================
# FILL REMAINING CUSTOMERS
# =========================================================

def _normalize_subproblem_coords(
    depot_xy: np.ndarray,
    customer_xy: np.ndarray,
) -> np.ndarray:
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
    nodes_data: Dict[int, Dict[str, Any]],
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
    nodes_data: Dict[int, Dict[str, Any]],
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
        [nodes_data[DEPOT_ID]["x"], nodes_data[DEPOT_ID]["y"]],
        dtype=np.float32,
    )

    customer_xy = np.array(
        [
            [nodes_data[v]["x"], nodes_data[v]["y"]]
            for v in remaining_vertices
        ],
        dtype=np.float32,
    )

    all_xy = _normalize_subproblem_coords(depot_xy, customer_xy)

    locs = torch.tensor(
        all_xy,
        dtype=torch.float32,
        device=_NEURAL_DEVICE,
    ).unsqueeze(0)

    demands = torch.tensor(
        [
            [
                nodes_data[v]["demand"] / capacity_value
                for v in remaining_vertices
            ]
        ],
        dtype=torch.float32,
        device=_NEURAL_DEVICE,
    )

    env = _make_env(
        num_nodes=len(remaining_vertices),
        capacity_value=capacity_value,
        device=_NEURAL_DEVICE,
    )

    env.reset(
        batch_size=1,
        locs=locs,
        demands=demands,
    )

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
    dimension: int,
    capacity: float,
    nodes: Dict[int, Dict[str, Any]],
    dist_matrix=None,
    nearest_neighbors=None,
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

    Lưu ý:
    - Không tự read_data trong GA.py nữa.
    - dimension, capacity, nodes, dist_matrix được truyền từ main.py.
    - dist_matrix dùng để tránh gọi euclid/sqrt lặp lại.
    """
    _ = nearest_neighbors  # GA chưa cần dùng KNN, giữ tham số để đồng bộ pipeline.

    resolved_ckpt_path = _resolve_ckpt_path(neural_ckpt_path)

    route_candidates = _collect_route_candidates(
        parent=parent,
        route=route,
        parent_indices=[par1, par2, par3],
        nodes_data=nodes,
        capacity_value=capacity,
        dist_matrix=dist_matrix,
        num_good_routes=num_good_routes_per_parent,
    )

    best_fitness = float("inf")
    best_child = None
    best_child_route = None

    warned_neural_failure = False

    all_vertices = set(range(2, dimension + 1))
    expected_len = dimension - 1

    for _trial in range(selection_trials):
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
                        print(
                            "[WARN] Neural fill failed. "
                            f"Fallback to random fill. Error: {e}"
                        )
                        warned_neural_failure = True

                    fallback_vertices = list(remaining_vertices)
                    random.shuffle(fallback_vertices)

                    fill_perm, fill_markers = _random_fill_remaining(
                        remaining_vertices=fallback_vertices,
                        nodes_data=nodes,
                        capacity_value=capacity,
                    )
            else:
                fallback_vertices = list(remaining_vertices)
                random.shuffle(fallback_vertices)

                fill_perm, fill_markers = _random_fill_remaining(
                    remaining_vertices=fallback_vertices,
                    nodes_data=nodes,
                    capacity_value=capacity,
                )

            child_permutation.extend(fill_perm)
            child_route_markers.extend(fill_markers)

        if len(child_permutation) != expected_len:
            raise ValueError(
                "Child permutation length mismatch: "
                f"got {len(child_permutation)}, expected {expected_len}"
            )

        if len(child_route_markers) != expected_len:
            raise ValueError(
                "Child route marker length mismatch: "
                f"got {len(child_route_markers)}, expected {expected_len}"
            )

        if len(set(child_permutation)) != expected_len:
            raise ValueError("Child has duplicated customers.")

        child_fitness = _solution_fitness(
            permutation=child_permutation,
            route_markers=child_route_markers,
            nodes_data=nodes,
            dist_matrix=dist_matrix,
        )

        if child_fitness < best_fitness:
            best_fitness = child_fitness
            best_child = child_permutation
            best_child_route = child_route_markers

    return best_child, best_child_route, best_fitness