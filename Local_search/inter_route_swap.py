from caculate import separate_routes
from Local_search.local_search_utils import (
    route_distance,
    route_demand,
    rebuild_solution,
    euclid,
    DEPOT_ID,
    ROUTE_PENALTY,
)


def _is_neighbor(a, b, nearest_neighbors):
    """
    Kiểm tra a và b có nằm trong tập K-nearest của nhau không.
    Dùng đối xứng để tránh bỏ sót quá nhiều candidate.
    """
    if nearest_neighbors is None:
        return True

    return (
        b in nearest_neighbors.get(a, set())
        or a in nearest_neighbors.get(b, set())
    )


def _swap_delta_between_routes(route1, i, route2, j, nodes, dist_matrix=None):
    """
    Delta distance O(1) khi swap:
        a = route1[i]
        b = route2[j]

    Route 1: prev1 - a - next1  ->  prev1 - b - next1
    Route 2: prev2 - b - next2  ->  prev2 - a - next2

    Vì r1 != r2 nên không có case cạnh chồng nhau trong cùng route.
    """
    a = route1[i]
    b = route2[j]

    prev1 = DEPOT_ID if i == 0 else route1[i - 1]
    next1 = DEPOT_ID if i == len(route1) - 1 else route1[i + 1]

    prev2 = DEPOT_ID if j == 0 else route2[j - 1]
    next2 = DEPOT_ID if j == len(route2) - 1 else route2[j + 1]

    old_r1 = (
        euclid(nodes, prev1, a, dist_matrix)
        + euclid(nodes, a, next1, dist_matrix)
    )
    new_r1 = (
        euclid(nodes, prev1, b, dist_matrix)
        + euclid(nodes, b, next1, dist_matrix)
    )

    old_r2 = (
        euclid(nodes, prev2, b, dist_matrix)
        + euclid(nodes, b, next2, dist_matrix)
    )
    new_r2 = (
        euclid(nodes, prev2, a, dist_matrix)
        + euclid(nodes, a, next2, dist_matrix)
    )

    delta_r1 = new_r1 - old_r1
    delta_r2 = new_r2 - old_r2

    return delta_r1 + delta_r2, delta_r1, delta_r2


def _passes_knn_filter(route1, i, route2, j, nearest_neighbors):
    """
    Lọc KNN nhẹ cho swap 1-1.

    Sau swap, các cạnh mới là:
        prev1-b, b-next1, prev2-a, a-next2

    Nếu không có cạnh mới nào liên quan đến node gần nhau thì bỏ qua.
    Điều này giảm số candidate nhưng vẫn ít aggressive hơn việc bắt buộc
    tất cả cạnh mới đều phải nằm trong KNN.
    """
    if nearest_neighbors is None:
        return True

    a = route1[i]
    b = route2[j]

    prev1 = DEPOT_ID if i == 0 else route1[i - 1]
    next1 = DEPOT_ID if i == len(route1) - 1 else route1[i + 1]

    prev2 = DEPOT_ID if j == 0 else route2[j - 1]
    next2 = DEPOT_ID if j == len(route2) - 1 else route2[j + 1]

    return (
        _is_neighbor(prev1, b, nearest_neighbors)
        or _is_neighbor(b, next1, nearest_neighbors)
        or _is_neighbor(prev2, a, nearest_neighbors)
        or _is_neighbor(a, next2, nearest_neighbors)
    )


def inter_route_swap(
    parent,
    route,
    fitness,
    nodes,
    capacity,
    dist_matrix=None,
    nearest_neighbors=None,
):
    """
    Swap 1-1 giữa hai route khác nhau.

    Bản tối ưu:
    - Dùng dist_matrix nếu được truyền vào.
    - Tính delta swap bằng công thức O(1), không rebuild route_distance
      cho từng candidate.
    - Có KNN filter nhẹ nếu nearest_neighbors được truyền vào.
    - First-improvement: gặp move tốt thì apply ngay và restart scan.
    """
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes, dist_matrix) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True

    while improved:
        improved = False

        for r1 in range(len(routes)):
            for r2 in range(r1 + 1, len(routes)):
                for i in range(len(routes[r1])):
                    a = routes[r1][i]
                    demand_a = nodes[a]["demand"]

                    for j in range(len(routes[r2])):
                        b = routes[r2][j]
                        demand_b = nodes[b]["demand"]

                        new_load_r1 = route_loads[r1] - demand_a + demand_b
                        new_load_r2 = route_loads[r2] - demand_b + demand_a

                        if new_load_r1 > capacity or new_load_r2 > capacity:
                            continue

                        if not _passes_knn_filter(
                            routes[r1],
                            i,
                            routes[r2],
                            j,
                            nearest_neighbors,
                        ):
                            continue

                        delta, delta_r1, delta_r2 = _swap_delta_between_routes(
                            routes[r1],
                            i,
                            routes[r2],
                            j,
                            nodes,
                            dist_matrix,
                        )

                        # Swap không đổi số route nên chỉ cần delta distance.
                        if delta < -1e-12:
                            routes[r1][i] = b
                            routes[r2][j] = a

                            route_costs[r1] += delta_r1
                            route_costs[r2] += delta_r2

                            route_loads[r1] = new_load_r1
                            route_loads[r2] = new_load_r2

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
