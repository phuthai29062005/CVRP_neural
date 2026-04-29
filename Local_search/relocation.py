from caculate import separate_routes
from Local_search.local_search_utils import (
    route_distance,
    route_demand,
    rebuild_solution,
    euclid,
    DEPOT_ID,
    ROUTE_PENALTY,
)


def _same_route_reloc_delta(route, i, j, nodes):
    """
    Delta cho relocation trong cùng route.
    Dùng route mới để tính route_distance vì công thức case cùng route dễ lỗi.
    Vẫn nhanh hơn rebuild toàn bộ nghiệm.
    """
    old_cost = route_distance(route, nodes)

    node = route[i]
    temp = route[:i] + route[i + 1:]
    temp = temp[:j] + [node] + temp[j:]

    new_cost = route_distance(temp, nodes)
    return temp, new_cost - old_cost


def _inter_route_reloc_delta(route1, i, route2, j, nodes):
    """
    Di chuyển route1[i] sang route2 tại vị trí j.
    Trả về delta distance O(1).
    """
    b = route1[i]
    a = DEPOT_ID if i == 0 else route1[i - 1]
    c = DEPOT_ID if i == len(route1) - 1 else route1[i + 1]

    u = DEPOT_ID if j == 0 else route2[j - 1]
    v = DEPOT_ID if j == len(route2) else route2[j]

    delta_remove = -euclid(nodes, a, b) - euclid(nodes, b, c) + euclid(nodes, a, c)
    delta_insert = -euclid(nodes, u, v) + euclid(nodes, u, b) + euclid(nodes, b, v)

    return delta_remove + delta_insert


def relocation(parent, route, fitness, nodes, capacity):
    """
    Relocation 1-customer, first improvement.
    Có check capacity và có xét giảm số route khi route nguồn rỗng.
    """
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True
    while improved:
        improved = False

        for r1 in range(len(routes)):
            for i in range(len(routes[r1])):
                node = routes[r1][i]
                demand = nodes[node]["demand"]

                for r2 in range(len(routes)):
                    # same-route relocation
                    if r1 == r2:
                        for j in range(len(routes[r1])):  # insert sau khi remove, nên max len-1 + 1 = len
                            if j == i:
                                continue

                            new_route_r, delta = _same_route_reloc_delta(routes[r1], i, j, nodes)

                            if delta < -1e-12:
                                routes[r1] = new_route_r
                                route_costs[r1] += delta
                                # load không đổi
                                improved = True
                                break

                        if improved:
                            break

                    else:
                        # capacity feasibility
                        if route_loads[r2] + demand > capacity:
                            continue

                        for j in range(len(routes[r2]) + 1):
                            delta = _inter_route_reloc_delta(routes[r1], i, routes[r2], j, nodes)

                            penalty_delta = 0.0
                            remove_empty_source = (len(routes[r1]) == 1)
                            if remove_empty_source:
                                penalty_delta -= ROUTE_PENALTY

                            total_delta = delta + penalty_delta

                            if total_delta < -1e-12:
                                node_to_move = routes[r1][i]

                                new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                                new_r2 = routes[r2][:j] + [node_to_move] + routes[r2][j:]

                                routes[r2] = new_r2
                                route_costs[r2] = route_distance(new_r2, nodes)
                                route_loads[r2] += demand

                                if remove_empty_source:
                                    del routes[r1]
                                    del route_costs[r1]
                                    del route_loads[r1]
                                else:
                                    routes[r1] = new_r1
                                    route_costs[r1] = route_distance(new_r1, nodes)
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