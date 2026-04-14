import os
import random
import numpy as np
import torch

from read_data import read_data
from caculate import get_good_routes, get_fitness
from Train_Neural.cvrp_model import CVRPModel
from Train_Neural.cvrp_env import CVRPenv


# =========================
# Instance loading
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "ML4VRP2026", "Instances", "cvrp", "vrp", "X-n101-k25.vrp")

dimension, capacity, nodes = read_data(FILE_PATH)
print(dimension, capacity)  # Kiểm tra dữ liệu đã đọc đúng chưa


# =========================
# Neural model cache
# =========================
_NEURAL_MODEL = None
_NEURAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NEURAL_LOAD_FAILED = False


def _resolve_ckpt_path(ckpt_path: str | None):
    """
    Resolve checkpoint path robustly:
    - absolute path: giữ nguyên
    - relative path: nối từ thư mục chứa GA.py
    """
    if ckpt_path is None:
        ckpt_path = os.path.join("checkpoints_neural_fill", "model_epoch_24.pt")

    if os.path.isabs(ckpt_path):
        return ckpt_path

    return os.path.join(BASE_DIR, ckpt_path)


def _load_neural_model(
    ckpt_path=None,
    embedding_dim=128,
    num_heads=8,
    num_layers=3,
):
    global _NEURAL_MODEL, _NEURAL_LOAD_FAILED

    if _NEURAL_MODEL is not None:
        return _NEURAL_MODEL

    if _NEURAL_LOAD_FAILED:
        raise FileNotFoundError("Neural model was already attempted and failed to load.")

    resolved_ckpt_path = _resolve_ckpt_path(ckpt_path)

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
    print(f"[INFO] Loaded neural model from: {resolved_ckpt_path}")
    return _NEURAL_MODEL


def get_vertices_from_routes(routes):
    """Lấy tất cả đỉnh từ danh sách routes."""
    vertices = set()
    for route in routes:
        for vertex in route:
            vertices.add(vertex)
    return vertices


def select_good_routes_random(good_routes, selection_prob=0.5):
    """Random chọn một số good routes."""
    count = 0
    selected_routes = []
    for route in good_routes:
        if random.random() < selection_prob:
            selected_routes.append(route)
            count += 1
        if count >= 5:  # Giới hạn số route được chọn tối đa là 5
            break
    return selected_routes


def remove_duplicate_vertices_in_routes(routes):
    """Loại bỏ routes có đỉnh trùng lặp."""
    kept_routes = []
    used_vertices = set()

    for route in routes:
        route_vertices = set(route)
        if route_vertices.isdisjoint(used_vertices):
            kept_routes.append(route)
            used_vertices.update(route_vertices)

    return kept_routes


def _normalize_subproblem_coords(depot_xy, customer_xy):
    """
    Normalize tọa độ về [0,1] theo bounding box của subproblem
    để gần distribution train của neural hơn.
    """
    all_xy = np.vstack([depot_xy[None, :], customer_xy]).astype(np.float32)

    min_xy = all_xy.min(axis=0, keepdims=True)
    max_xy = all_xy.max(axis=0, keepdims=True)
    scale = np.maximum(max_xy - min_xy, 1e-8)

    all_xy = (all_xy - min_xy) / scale
    return all_xy


def _random_fill_remaining(remaining_vertices, nodes, capacity):
    """
    Fallback random fill có capacity check.
    remaining_vertices nên đã được sort/shuffle trước khi vào đây.
    """
    perm = []
    markers = []

    current_load = 0
    for idx_v, vertex in enumerate(remaining_vertices):
        demand = nodes[vertex]["demand"]

        if idx_v == 0:
            perm.append(vertex)
            markers.append(1)
            current_load = demand
        elif current_load + demand <= capacity:
            perm.append(vertex)
            markers.append(0)
            current_load += demand
        else:
            perm.append(vertex)
            markers.append(1)
            current_load = demand

    return perm, markers


@torch.no_grad()
def solve_remaining_with_neural(
    remaining_vertices,
    nodes,
    capacity,
    ckpt_path=None,
    decode_type="greedy",
    use_vehicle_penalty=False,
    vehicle_penalty=0.0,
):
    """
    Dùng neural để giải residual CVRP trên remaining_vertices.

    Returns:
        neural_permutation: [v1, v2, ...]
        neural_route_markers: [1,0,0,1,0,...]
    """
    if len(remaining_vertices) == 0:
        return [], []

    # Thứ tự phải ổn định để local action i -> global vertex remaining_vertices[i-1] không bị lệch
    remaining_vertices = sorted(remaining_vertices)

    model = _load_neural_model(ckpt_path=ckpt_path)

    depot_xy = np.array([nodes[1]["x"], nodes[1]["y"]], dtype=np.float32)
    customer_xy = np.array(
        [[nodes[v]["x"], nodes[v]["y"]] for v in remaining_vertices],
        dtype=np.float32,
    )

    all_xy = _normalize_subproblem_coords(depot_xy, customer_xy)
    locs = torch.tensor(all_xy, dtype=torch.float32, device=_NEURAL_DEVICE).unsqueeze(0)
    # locs: [1, m+1, 2]

    demands = torch.tensor(
        [[nodes[v]["demand"] / capacity for v in remaining_vertices]],
        dtype=torch.float32,
        device=_NEURAL_DEVICE,
    )
    # demands: [1, m]

    env = CVRPenv(
        num_nodes=len(remaining_vertices),
        capacity=capacity,
        device=_NEURAL_DEVICE,
        vehicle_penalty=vehicle_penalty,
        use_vehicle_penalty=use_vehicle_penalty,
    )
    env.reset(batch_size=1, locs=locs, demands=demands)

    state = env.get_state()
    embeddings = model._get_embeddings(state["locs"], state["demands"])

    done = torch.zeros(1, dtype=torch.bool, device=_NEURAL_DEVICE)
    action_seq = []

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

        action_seq.append(action.item())
        state, _, done = env.step(action)

    # Convert local actions -> global vertices
    # action 0 = depot
    # action i>0 => remaining_vertices[i-1]
    neural_permutation = []
    neural_route_markers = []

    start_new_route = True
    for a in action_seq:
        if a == 0:
            start_new_route = True
        else:
            global_vertex = remaining_vertices[a - 1]
            neural_permutation.append(global_vertex)
            neural_route_markers.append(1 if start_new_route else 0)
            start_new_route = False

    return neural_permutation, neural_route_markers


def _flatten_kept_routes(kept_routes):
    """
    kept_routes = [[...], [...], ...]
    -> permutation + route_markers
    """
    child_permutation = []
    child_route_markers = []

    for route_seg in kept_routes:
        for j, vertex in enumerate(route_seg):
            child_permutation.append(vertex)
            child_route_markers.append(1 if j == 0 else 0)

    return child_permutation, child_route_markers


def GA(
    parent,
    route,
    par1,
    par2,
    par3,
    use_neural_fill=True,
    neural_ckpt_path=os.path.join("checkpoints_neural_fill", "model_epoch_24.pt"),
    neural_decode_type="greedy",
):
    """
    GA Crossover sử dụng good routes + neural fill cho remaining_vertices

    Returns:
        (best_child_permutation, best_child_route, best_child_fitness)
    """
    resolved_ckpt_path = _resolve_ckpt_path(neural_ckpt_path)

    all_good_routes = []
    for idx in [par1, par2, par3]:
        good_routes, _ = get_good_routes(parent[idx], route[idx], nodes, num_good_routes=5)
        all_good_routes.extend(good_routes)

    best_fitness = float("inf")
    best_child = None
    best_child_route = None

    warn_once = False

    for _ in range(5):
        selected_routes = select_good_routes_random(all_good_routes, selection_prob=0.5)
        kept_routes = remove_duplicate_vertices_in_routes(selected_routes)

        used_vertices = get_vertices_from_routes(kept_routes)

        all_vertices = set(range(2, dimension + 1))
        remaining_vertices = sorted(list(all_vertices - used_vertices))

        child_permutation, child_route_markers = _flatten_kept_routes(kept_routes)

        if len(remaining_vertices) > 0:
            if use_neural_fill:
                try:
                    neural_perm, neural_markers = solve_remaining_with_neural(
                        remaining_vertices=remaining_vertices,
                        nodes=nodes,
                        capacity=capacity,
                        ckpt_path=resolved_ckpt_path,
                        decode_type=neural_decode_type,
                        use_vehicle_penalty=False,
                        vehicle_penalty=0.0,
                    )
                except Exception as e:
                    if not warn_once:
                        print(f"[WARN] Neural fill failed once, fallback to random fill. Error: {e}")
                        warn_once = True

                    fallback_vertices = remaining_vertices.copy()
                    random.shuffle(fallback_vertices)
                    neural_perm, neural_markers = _random_fill_remaining(
                        fallback_vertices, nodes, capacity
                    )
            else:
                fallback_vertices = remaining_vertices.copy()
                random.shuffle(fallback_vertices)
                neural_perm, neural_markers = _random_fill_remaining(
                    fallback_vertices, nodes, capacity
                )

            child_permutation.extend(neural_perm)
            child_route_markers.extend(neural_markers)

        if len(child_permutation) != dimension - 1:
            raise ValueError(
                f"Child permutation length mismatch: got {len(child_permutation)}, "
                f"expected {dimension - 1}"
            )

        if len(child_route_markers) != dimension - 1:
            raise ValueError(
                f"Child route marker length mismatch: got {len(child_route_markers)}, "
                f"expected {dimension - 1}"
            )

        child_fitness = get_fitness(child_permutation, child_route_markers, nodes)

        if child_fitness < best_fitness:
            best_fitness = child_fitness
            best_child = child_permutation
            best_child_route = child_route_markers

    return best_child, best_child_route, best_fitness