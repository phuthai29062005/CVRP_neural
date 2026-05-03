from itertools import permutations

from caculate import separate_routes
from Local_search.local_search_utils import (
    route_demand,
    rebuild_solution,
    DEPOT_ID,
    ROUTE_PENALTY,
)

EPS = 1e-12

BASE_MAX_SOURCE_ROUTE_SIZE = 3
MAX_DYNAMIC_SOURCE_ROUTE_SIZE = 5
TOP_INSERTIONS_PER_CUSTOMER = 8


def _dist(nodes, a, b, dist_matrix=None):
    """Distance helper. Ưu tiên dist_matrix nếu đã precompute."""
    if dist_matrix is not None:
        return float(dist_matrix[a, b])

    ax, ay = nodes[a]["x"], nodes[a]["y"]
    bx, by = nodes[b]["x"], nodes[b]["y"]

    dx = ax - bx
    dy = ay - by

    return (dx * dx + dy * dy) ** 0.5


def _route_distance(route, nodes, dist_matrix=None):
    """Distance của 1 route: depot -> customers -> depot."""
    if not route:
        return 0.0

    total = _dist(nodes, DEPOT_ID, route[0], dist_matrix)

    for i in range(len(route) - 1):
        total += _dist(nodes, route[i], route[i + 1], dist_matrix)

    total += _dist(nodes, route[-1], DEPOT_ID, dist_matrix)

    return total


def _solution_fitness(routes, nodes, dist_matrix=None):
    """F = 1000 * number_of_routes + total_distance."""
    return len(routes) * ROUTE_PENALTY + sum(
        _route_distance(r, nodes, dist_matrix)
        for r in routes
    )


def get_dynamic_max_source_route_size(
    generation=None,
    renew_count=0,
    base_size=BASE_MAX_SOURCE_ROUTE_SIZE,
    max_size=MAX_DYNAMIC_SOURCE_ROUTE_SIZE,
):
    """
    MAX_SOURCE_ROUTE_SIZE động.

    Gợi ý mặc định:
    - Đầu search: chỉ xóa route 1-3 customer để nhanh.
    - Sau generation >= 30: cho thử route 4 customer.
    - Sau generation >= 60 hoặc renew nhiều: cho thử route 5 customer.

    Có thể chỉnh ngưỡng này tùy runtime.
    """
    size = int(base_size)

    if generation is not None:
        if generation >= 30:
            size += 1
        if generation >= 60:
            size += 1

    if renew_count >= 2:
        size = max(size, base_size + 1)
    if renew_count >= 4:
        size = max(size, base_size + 2)

    return max(1, min(size, max_size))


def _is_neighbor_insert(customer, left, right, nearest_neighbors):
    """
    Lọc vị trí insert bằng K-nearest.
    Nếu KNN quá chặt và không còn candidate, hàm gọi sẽ fallback không KNN.
    """
    if nearest_neighbors is None:
        return True

    customer_neighbors = nearest_neighbors.get(customer, set())
    left_neighbors = nearest_neighbors.get(left, set())
    right_neighbors = nearest_neighbors.get(right, set())

    return (
        left in customer_neighbors
        or right in customer_neighbors
        or customer in left_neighbors
        or customer in right_neighbors
    )


def _insert_delta(route, customer, pos, nodes, dist_matrix=None):
    """Delta distance O(1) khi chèn customer vào route tại vị trí pos."""
    prev_node = DEPOT_ID if pos == 0 else route[pos - 1]
    next_node = DEPOT_ID if pos == len(route) else route[pos]

    return (
        -_dist(nodes, prev_node, next_node, dist_matrix)
        +_dist(nodes, prev_node, customer, dist_matrix)
        +_dist(nodes, customer, next_node, dist_matrix)
    )


def _get_top_insertions(
    routes,
    route_loads,
    customer,
    nodes,
    capacity,
    dist_matrix=None,
    nearest_neighbors=None,
    top_k=TOP_INSERTIONS_PER_CUSTOMER,
):
    """
    Tìm TOP_K vị trí chèn tốt nhất cho một customer.
    """
    demand = nodes[customer]["demand"]
    candidates = []

    for r_idx, r in enumerate(routes):
        if route_loads[r_idx] + demand > capacity:
            continue

        for pos in range(len(r) + 1):
            prev_node = DEPOT_ID if pos == 0 else r[pos - 1]
            next_node = DEPOT_ID if pos == len(r) else r[pos]

            if not _is_neighbor_insert(customer, prev_node, next_node, nearest_neighbors):
                continue

            delta = _insert_delta(r, customer, pos, nodes, dist_matrix)
            candidates.append((delta, r_idx, pos))

    if not candidates and nearest_neighbors is not None:
        # fallback an toàn nếu KNN lọc hết candidate
        for r_idx, r in enumerate(routes):
            if route_loads[r_idx] + demand > capacity:
                continue

            for pos in range(len(r) + 1):
                delta = _insert_delta(r, customer, pos, nodes, dist_matrix)
                candidates.append((delta, r_idx, pos))

    candidates.sort(key=lambda x: x[0])

    return candidates[:top_k]


def _try_eliminate_small_route(
    routes,
    source_idx,
    nodes,
    capacity,
    dist_matrix=None,
    nearest_neighbors=None,
    max_source_route_size=BASE_MAX_SOURCE_ROUTE_SIZE,
):
    """
    Thử xóa một route nhỏ bằng cách chuyển toàn bộ customer sang các route khác.

    max_source_route_size được truyền động từ generation/renew_count.
    """
    source_route = list(routes[source_idx])

    if not (1 <= len(source_route) <= max_source_route_size):
        return False, routes, float("inf")

    base_routes = [
        list(r)
        for idx, r in enumerate(routes)
        if idx != source_idx
    ]
    base_loads = [route_demand(r, nodes) for r in base_routes]

    best_routes = None
    best_fitness = float("inf")

    for order in set(permutations(source_route)):
        order = list(order)

        def dfs(k, cur_routes, cur_loads):
            nonlocal best_routes, best_fitness

            if k == len(order):
                fit = _solution_fitness(cur_routes, nodes, dist_matrix)

                if fit < best_fitness:
                    best_fitness = fit
                    best_routes = [list(r) for r in cur_routes]

                return

            customer = order[k]
            demand = nodes[customer]["demand"]

            insertions = _get_top_insertions(
                routes=cur_routes,
                route_loads=cur_loads,
                customer=customer,
                nodes=nodes,
                capacity=capacity,
                dist_matrix=dist_matrix,
                nearest_neighbors=nearest_neighbors,
                top_k=TOP_INSERTIONS_PER_CUSTOMER,
            )

            for _, r_idx, pos in insertions:
                new_routes = [list(r) for r in cur_routes]
                new_loads = list(cur_loads)

                new_routes[r_idx].insert(pos, customer)
                new_loads[r_idx] += demand

                dfs(k + 1, new_routes, new_loads)

        dfs(0, base_routes, base_loads)

    if best_routes is None:
        return False, routes, float("inf")

    return True, best_routes, best_fitness


def route_elimination(
    parent,
    route,
    fitness,
    nodes,
    capacity,
    dist_matrix=None,
    nearest_neighbors=None,
    generation=None,
    renew_count=0,
    max_source_route_size=None,
):
    """
    Best-improvement route elimination.

    Sửa chính:
    - MAX_SOURCE_ROUTE_SIZE không còn cứng = 3.
    - Có thể tăng dần theo generation hoặc renew_count.
    - Vẫn ưu tiên route nhỏ trước để giữ runtime ổn định.
    """
    routes = separate_routes(parent, route)

    if max_source_route_size is None:
        max_source_route_size = get_dynamic_max_source_route_size(
            generation=generation,
            renew_count=renew_count,
        )

    improved = True

    while improved:
        improved = False

        current_fitness = _solution_fitness(routes, nodes, dist_matrix)

        best_candidate_routes = None
        best_candidate_fitness = current_fitness

        candidate_indices = sorted(
            [
                idx
                for idx, r in enumerate(routes)
                if 1 <= len(r) <= max_source_route_size
            ],
            key=lambda idx: (
                len(routes[idx]),
                route_demand(routes[idx], nodes),
            ),
        )

        for source_idx in candidate_indices:
            success, new_routes, new_fitness = _try_eliminate_small_route(
                routes=routes,
                source_idx=source_idx,
                nodes=nodes,
                capacity=capacity,
                dist_matrix=dist_matrix,
                nearest_neighbors=nearest_neighbors,
                max_source_route_size=max_source_route_size,
            )

            if not success:
                continue

            if new_fitness < best_candidate_fitness - EPS:
                best_candidate_fitness = new_fitness
                best_candidate_routes = new_routes

        if best_candidate_routes is not None:
            routes = best_candidate_routes
            improved = True

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = _solution_fitness(routes, nodes, dist_matrix)

    return new_parent, new_route, new_fitness
