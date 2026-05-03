from caculate import separate_routes
from Local_search.local_search_utils import route_distance, rebuild_solution, euclid, DEPOT_ID


def opt_2(parent, route, fitness, nodes, dist_matrix=None, nearest_neighbors=None):
    """
    Intra-route 2-opt với first improvement + delta O(1).
    Fitness tổng thể không đổi số route, nên chỉ cần xét delta distance trong route.
    """
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes, dist_matrix) for r in routes]

    improved = True
    while improved:
        improved = False

        for r_idx, customers in enumerate(routes):
            n = len(customers)
            if n < 3:
                continue

            old_cost = route_costs[r_idx]

            for i in range(n - 1):
                a = DEPOT_ID if i == 0 else customers[i - 1]
                b = customers[i]

                for j in range(i + 1, n):
                    c = customers[j]
                    d = DEPOT_ID if j == n - 1 else customers[j + 1]

                    # bỏ move vô nghĩa khi cắt cạnh liền nhau ở giữa route vẫn cho phép,
                    # nhưng delta vẫn đúng nên không cần skip riêng
                    if nearest_neighbors is not None:
                        # 2-opt tốt thường tạo cạnh mới giữa các node gần nhau.
                        if (
                            c not in nearest_neighbors.get(a, set())
                            and a not in nearest_neighbors.get(c, set())
                            and d not in nearest_neighbors.get(b, set())
                            and b not in nearest_neighbors.get(d, set())
                        ):
                            continue

                    delta = (
                        -euclid(nodes, a, b, dist_matrix)
                        -euclid(nodes, c, d, dist_matrix)
                        + euclid(nodes, a, c, dist_matrix)
                        + euclid(nodes, b, d, dist_matrix)
                    )

                    if delta < -1e-12:
                        # apply first improvement
                        customers[i:j + 1] = reversed(customers[i:j + 1])
                        route_costs[r_idx] = old_cost + delta
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * 1000.0 + sum(route_costs)
    return new_parent, new_route, new_fitness