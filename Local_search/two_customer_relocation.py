from itertools import combinations

from caculate import separate_routes
from Local_search.local_search_utils import (
    route_distance,
    route_demand,
    rebuild_solution,
    ROUTE_PENALTY,
)


def _best_insert_two(target_route, customers, nodes):
    """
    Tìm cách chèn 2 customer vào target_route cho distance nhỏ nhất.
    customers: tuple/list gồm 2 customer.
    """
    c1, c2 = customers

    best_route = None
    best_cost = float("inf")

    # Thử cả hai thứ tự c1-c2 và c2-c1
    for order in [(c1, c2), (c2, c1)]:
        # Insert customer thứ nhất
        for pos1 in range(len(target_route) + 1):
            temp1 = list(target_route)
            temp1.insert(pos1, order[0])

            # Insert customer thứ hai
            for pos2 in range(len(temp1) + 1):
                temp2 = list(temp1)
                temp2.insert(pos2, order[1])

                cost = route_distance(temp2, nodes)

                if cost < best_cost:
                    best_cost = cost
                    best_route = temp2

    return best_route, best_cost


def two_customer_relocation(parent, route, fitness, nodes, capacity):
    """
    Di chuyển 2 customer cùng lúc từ route r1 sang route r2.
    Có thể giúp làm rỗng/giảm tải một route mà 1-customer relocation không làm được.
    """
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True

    while improved:
        improved = False

        for r1 in range(len(routes)):
            if len(routes[r1]) < 2:
                continue

            for i, j in combinations(range(len(routes[r1])), 2):
                customers_to_move = (routes[r1][i], routes[r1][j])
                move_demand = nodes[customers_to_move[0]]["demand"] + nodes[customers_to_move[1]]["demand"]

                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue

                    if route_loads[r2] + move_demand > capacity:
                        continue

                    new_r1 = [
                        c for idx, c in enumerate(routes[r1])
                        if idx not in (i, j)
                    ]

                    new_r2, new_r2_cost = _best_insert_two(
                        routes[r2],
                        customers_to_move,
                        nodes
                    )

                    old_route_count = len(routes)
                    new_route_count = old_route_count

                    if len(new_r1) == 0:
                        new_route_count -= 1

                    old_cost = (
                        old_route_count * ROUTE_PENALTY
                        + sum(route_costs)
                    )

                    new_cost_sum = 0.0
                    for k in range(len(routes)):
                        if k == r1:
                            if len(new_r1) > 0:
                                new_cost_sum += route_distance(new_r1, nodes)
                        elif k == r2:
                            new_cost_sum += new_r2_cost
                        else:
                            new_cost_sum += route_costs[k]

                    new_fitness = new_route_count * ROUTE_PENALTY + new_cost_sum

                    if new_fitness < old_cost - 1e-12:
                        routes[r2] = new_r2
                        route_costs[r2] = route_distance(new_r2, nodes)
                        route_loads[r2] += move_demand

                        if len(new_r1) == 0:
                            del routes[r1]
                            del route_costs[r1]
                            del route_loads[r1]
                        else:
                            routes[r1] = new_r1
                            route_costs[r1] = route_distance(new_r1, nodes)
                            route_loads[r1] -= move_demand

                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * ROUTE_PENALTY + sum(route_distance(r, nodes) for r in routes)

    return new_parent, new_route, new_fitness