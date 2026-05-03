from caculate import separate_routes
from Local_search.local_search_utils import (
    route_demand,
    rebuild_solution,
    DEPOT_ID,
    ROUTE_PENALTY,
)

EPS = 1e-12


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


def _is_neighbor_move(node, left, right, nearest_neighbors):
    """
    Lọc candidate bằng K-nearest.
    Cho phép nếu node gần left/right hoặc left/right gần node.
    """
    if nearest_neighbors is None:
        return True

    node_neighbors = nearest_neighbors.get(node, set())
    left_neighbors = nearest_neighbors.get(left, set())
    right_neighbors = nearest_neighbors.get(right, set())

    return (
        left in node_neighbors
        or right in node_neighbors
        or node in left_neighbors
        or node in right_neighbors
    )


def _node_after_removal(route, remove_idx, pos):
    """
    Lấy node tại vị trí pos trong route sau khi bỏ remove_idx.
    route_after = route[:remove_idx] + route[remove_idx + 1:]
    Hàm này tránh tạo route_after khi chỉ cần tính delta.
    """
    if pos < remove_idx:
        return route[pos]

    return route[pos + 1]


def _same_route_reloc_delta(route, i, j, nodes, dist_matrix=None):
    """
    Delta O(1) cho same-route relocation.

    Move:
        node = route[i]
    Remove node trước, rồi insert lại vào vị trí j của route sau khi remove.

    j chạy từ 0 đến len(route)-1 vì route sau remove có len = n-1,
    số vị trí insert hợp lệ là n.

    Return:
        delta_distance
    """
    n = len(route)

    if n <= 1 or j == i:
        return 0.0

    node = route[i]

    # Cạnh bị ảnh hưởng khi remove node khỏi vị trí cũ
    a = DEPOT_ID if i == 0 else route[i - 1]
    c = DEPOT_ID if i == n - 1 else route[i + 1]

    delta_remove = (
        -_dist(nodes, a, node, dist_matrix)
        -_dist(nodes, node, c, dist_matrix)
        +_dist(nodes, a, c, dist_matrix)
    )

    # Cạnh bị ảnh hưởng khi insert node vào vị trí j trong route sau remove
    # route_after có length n-1, insert positions: 0..n-1
    if j == 0:
        u = DEPOT_ID
    else:
        u = _node_after_removal(route, i, j - 1)

    if j == n - 1:
        v = DEPOT_ID
    else:
        v = _node_after_removal(route, i, j)

    delta_insert = (
        -_dist(nodes, u, v, dist_matrix)
        +_dist(nodes, u, node, dist_matrix)
        +_dist(nodes, node, v, dist_matrix)
    )

    return delta_remove + delta_insert


def _apply_same_route_relocation(route, i, j):
    """
    Apply same-route relocation.
    j là vị trí insert trong route sau khi remove i.
    """
    node = route[i]
    new_route = route[:i] + route[i + 1:]
    new_route = new_route[:j] + [node] + new_route[j:]

    return new_route


def _inter_route_reloc_delta(route1, i, route2, j, nodes, dist_matrix=None):
    """
    Delta O(1) khi di chuyển route1[i] sang route2 tại vị trí j.
    """
    b = route1[i]

    a = DEPOT_ID if i == 0 else route1[i - 1]
    c = DEPOT_ID if i == len(route1) - 1 else route1[i + 1]

    u = DEPOT_ID if j == 0 else route2[j - 1]
    v = DEPOT_ID if j == len(route2) else route2[j]

    delta_remove = (
        -_dist(nodes, a, b, dist_matrix)
        -_dist(nodes, b, c, dist_matrix)
        +_dist(nodes, a, c, dist_matrix)
    )

    delta_insert = (
        -_dist(nodes, u, v, dist_matrix)
        +_dist(nodes, u, b, dist_matrix)
        +_dist(nodes, b, v, dist_matrix)
    )

    return delta_remove + delta_insert


def relocation(
    parent,
    route,
    fitness,
    nodes,
    capacity,
    dist_matrix=None,
    nearest_neighbors=None,
):
    """
    Relocation 1-customer, first improvement.

    Sửa chính:
    - Same-route relocation dùng delta O(1), không gọi route_distance 2 lần O(n).
    - Inter-route relocation vẫn dùng delta O(1).
    - Có thể dùng nearest_neighbors để lọc bớt candidate insert.
    """
    routes = separate_routes(parent, route)
    route_costs = [_route_distance(r, nodes, dist_matrix) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True

    while improved:
        improved = False

        for r1 in range(len(routes)):
            for i in range(len(routes[r1])):
                node = routes[r1][i]
                demand = nodes[node]["demand"]

                for r2 in range(len(routes)):
                    # =====================================================
                    # Same-route relocation
                    # =====================================================
                    if r1 == r2:
                        n = len(routes[r1])

                        if n <= 2:
                            continue

                        for j in range(n):
                            if j == i:
                                continue

                            # Lọc KNN cho vị trí insert mới
                            if nearest_neighbors is not None:
                                if j == 0:
                                    u = DEPOT_ID
                                else:
                                    u = _node_after_removal(routes[r1], i, j - 1)

                                if j == n - 1:
                                    v = DEPOT_ID
                                else:
                                    v = _node_after_removal(routes[r1], i, j)

                                if not _is_neighbor_move(node, u, v, nearest_neighbors):
                                    continue

                            delta = _same_route_reloc_delta(
                                routes[r1],
                                i,
                                j,
                                nodes,
                                dist_matrix,
                            )

                            if delta < -EPS:
                                routes[r1] = _apply_same_route_relocation(
                                    routes[r1],
                                    i,
                                    j,
                                )
                                route_costs[r1] += delta
                                improved = True
                                break

                        if improved:
                            break

                    # =====================================================
                    # Inter-route relocation
                    # =====================================================
                    else:
                        if route_loads[r2] + demand > capacity:
                            continue

                        for j in range(len(routes[r2]) + 1):
                            u = DEPOT_ID if j == 0 else routes[r2][j - 1]
                            v = DEPOT_ID if j == len(routes[r2]) else routes[r2][j]

                            if not _is_neighbor_move(node, u, v, nearest_neighbors):
                                continue

                            delta = _inter_route_reloc_delta(
                                routes[r1],
                                i,
                                routes[r2],
                                j,
                                nodes,
                                dist_matrix,
                            )

                            remove_empty_source = len(routes[r1]) == 1
                            penalty_delta = -ROUTE_PENALTY if remove_empty_source else 0.0

                            total_delta = delta + penalty_delta

                            if total_delta < -EPS:
                                node_to_move = routes[r1][i]

                                new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                                new_r2 = routes[r2][:j] + [node_to_move] + routes[r2][j:]

                                routes[r2] = new_r2
                                route_costs[r2] = _route_distance(new_r2, nodes, dist_matrix)
                                route_loads[r2] += demand

                                if remove_empty_source:
                                    del routes[r1]
                                    del route_costs[r1]
                                    del route_loads[r1]
                                else:
                                    routes[r1] = new_r1
                                    route_costs[r1] = _route_distance(new_r1, nodes, dist_matrix)
                                    route_loads[r1] -= demand

                                improved = True
                                break

                        if improved:
                            break

                if improved:
                    break

            if improved:
                break

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * ROUTE_PENALTY + sum(route_costs)

    return new_parent, new_route, new_fitness
