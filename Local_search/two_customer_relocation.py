from itertools import combinations

from caculate import separate_routes
from Local_search.local_search_utils import (
    route_demand,
    rebuild_solution,
    DEPOT_ID,
    ROUTE_PENALTY,
)

EPS = 1e-12
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


def _is_neighbor_insert(customer, left, right, nearest_neighbors):
    """
    Lọc candidate bằng K-nearest.
    Cho phép nếu customer gần left/right hoặc ngược lại.
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
    """Delta O(1) khi insert customer vào route tại pos."""
    left = DEPOT_ID if pos == 0 else route[pos - 1]
    right = DEPOT_ID if pos == len(route) else route[pos]

    return (
        -_dist(nodes, left, right, dist_matrix)
        +_dist(nodes, left, customer, dist_matrix)
        +_dist(nodes, customer, right, dist_matrix)
    )


def _top_insertions(
    route,
    customer,
    nodes,
    dist_matrix=None,
    nearest_neighbors=None,
    top_k=TOP_INSERTIONS_PER_CUSTOMER,
):
    """
    Lấy TOP_K vị trí insert tốt nhất cho 1 customer.
    Nếu KNN lọc hết candidate thì fallback quét không KNN để tránh mất nghiệm.
    """
    candidates = []

    for pos in range(len(route) + 1):
        left = DEPOT_ID if pos == 0 else route[pos - 1]
        right = DEPOT_ID if pos == len(route) else route[pos]

        if not _is_neighbor_insert(customer, left, right, nearest_neighbors):
            continue

        delta = _insert_delta(route, customer, pos, nodes, dist_matrix)
        candidates.append((delta, pos))

    if not candidates and nearest_neighbors is not None:
        # fallback an toàn nếu KNN quá chặt
        for pos in range(len(route) + 1):
            delta = _insert_delta(route, customer, pos, nodes, dist_matrix)
            candidates.append((delta, pos))

    candidates.sort(key=lambda x: x[0])

    return candidates[:top_k]


def _best_insert_two(
    target_route,
    customers,
    nodes,
    dist_matrix=None,
    nearest_neighbors=None,
    top_k=TOP_INSERTIONS_PER_CUSTOMER,
):
    """
    Tìm cách chèn 2 customer vào target_route.

    Sửa chính:
    - Không thử mọi cặp vị trí O(n^2) nữa.
    - Với customer thứ nhất: chỉ giữ TOP_K vị trí tốt nhất.
    - Với customer thứ hai: trên mỗi route tạm, chỉ giữ TOP_K vị trí tốt nhất.

    Complexity xấp xỉ O(2 * n + 2 * TOP_K * n), thay vì O(n^2).
    """
    c1, c2 = customers

    best_route = None
    best_cost = float("inf")

    for order in [(c1, c2), (c2, c1)]:
        first_customer, second_customer = order

        first_insertions = _top_insertions(
            route=target_route,
            customer=first_customer,
            nodes=nodes,
            dist_matrix=dist_matrix,
            nearest_neighbors=nearest_neighbors,
            top_k=top_k,
        )

        for _, pos1 in first_insertions:
            temp1 = list(target_route)
            temp1.insert(pos1, first_customer)

            second_insertions = _top_insertions(
                route=temp1,
                customer=second_customer,
                nodes=nodes,
                dist_matrix=dist_matrix,
                nearest_neighbors=nearest_neighbors,
                top_k=top_k,
            )

            for _, pos2 in second_insertions:
                temp2 = list(temp1)
                temp2.insert(pos2, second_customer)

                cost = _route_distance(temp2, nodes, dist_matrix)

                if cost < best_cost:
                    best_cost = cost
                    best_route = temp2

    return best_route, best_cost


def two_customer_relocation(
    parent,
    route,
    fitness,
    nodes,
    capacity,
    dist_matrix=None,
    nearest_neighbors=None,
    top_k=TOP_INSERTIONS_PER_CUSTOMER,
):
    """
    Di chuyển 2 customer cùng lúc từ route r1 sang route r2.

    Sửa chính:
    - _best_insert_two không còn thử mọi vị trí O(n^2).
    - Dùng TOP_K vị trí insert tốt nhất cho từng customer.
    - Có thể dùng nearest_neighbors để lọc candidate.
    """
    routes = separate_routes(parent, route)
    route_costs = [_route_distance(r, nodes, dist_matrix) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True

    while improved:
        improved = False

        for r1 in range(len(routes)):
            if len(routes[r1]) < 2:
                continue

            for i, j in combinations(range(len(routes[r1])), 2):
                customers_to_move = (routes[r1][i], routes[r1][j])
                move_demand = (
                    nodes[customers_to_move[0]]["demand"]
                    + nodes[customers_to_move[1]]["demand"]
                )

                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue

                    if route_loads[r2] + move_demand > capacity:
                        continue

                    new_r1 = [
                        c
                        for idx, c in enumerate(routes[r1])
                        if idx not in (i, j)
                    ]

                    new_r2, new_r2_cost = _best_insert_two(
                        target_route=routes[r2],
                        customers=customers_to_move,
                        nodes=nodes,
                        dist_matrix=dist_matrix,
                        nearest_neighbors=nearest_neighbors,
                        top_k=top_k,
                    )

                    if new_r2 is None:
                        continue

                    old_route_count = len(routes)
                    new_route_count = old_route_count

                    if len(new_r1) == 0:
                        new_route_count -= 1

                    old_fitness = old_route_count * ROUTE_PENALTY + sum(route_costs)

                    new_cost_sum = 0.0

                    for k in range(len(routes)):
                        if k == r1:
                            if len(new_r1) > 0:
                                new_cost_sum += _route_distance(new_r1, nodes, dist_matrix)
                        elif k == r2:
                            new_cost_sum += new_r2_cost
                        else:
                            new_cost_sum += route_costs[k]

                    new_fitness = new_route_count * ROUTE_PENALTY + new_cost_sum

                    if new_fitness < old_fitness - EPS:
                        routes[r2] = new_r2
                        route_costs[r2] = new_r2_cost
                        route_loads[r2] += move_demand

                        if len(new_r1) == 0:
                            del routes[r1]
                            del route_costs[r1]
                            del route_loads[r1]
                        else:
                            routes[r1] = new_r1
                            route_costs[r1] = _route_distance(new_r1, nodes, dist_matrix)
                            route_loads[r1] -= move_demand

                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * ROUTE_PENALTY + sum(route_costs)

    return new_parent, new_route, new_fitness
