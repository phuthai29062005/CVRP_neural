from itertools import combinations

from caculate import separate_routes
from Local_search.local_search_utils import (
    route_distance,
    route_demand,
    rebuild_solution,
    ROUTE_PENALTY,
)


EPS = 1e-12
MAX_PAIRS = 50



def _limited_pairs(n, max_pairs=MAX_PAIRS):
    """
    Sinh tối đa max_pairs cặp index để tránh duyệt toàn bộ combinations.
    Dùng deterministic first pairs để kết quả chạy ổn định giữa các lần.
    """
    count = 0
    for pair in combinations(range(n), 2):
        yield pair
        count += 1
        if count >= max_pairs:
            break

def _remove_indices(route, indices):
    """
    Xóa các vị trí trong route.
    indices: iterable index cần xóa.
    """
    remove_set = set(indices)
    return [c for idx, c in enumerate(route) if idx not in remove_set]


def _sequence_demand(seq, nodes):
    return sum(nodes[c]["demand"] for c in seq)


def _best_insert_sequence(base_route, seq, nodes):
    """
    Chèn 1 hoặc 2 customer vào base_route sao cho route_distance nhỏ nhất.

    seq có thể là:
        [a] hoặc [a, b]

    Với 2 customer, thử cả hai thứ tự:
        a-b và b-a
    """
    if len(seq) == 0:
        return list(base_route), route_distance(base_route, nodes)

    best_route = None
    best_cost = float("inf")

    if len(seq) == 1:
        c = seq[0]

        for pos in range(len(base_route) + 1):
            temp = list(base_route)
            temp.insert(pos, c)

            cost = route_distance(temp, nodes)

            if cost < best_cost:
                best_cost = cost
                best_route = temp

    elif len(seq) == 2:
        a, b = seq

        for order in [(a, b), (b, a)]:
            for pos1 in range(len(base_route) + 1):
                temp1 = list(base_route)
                temp1.insert(pos1, order[0])

                for pos2 in range(len(temp1) + 1):
                    temp2 = list(temp1)
                    temp2.insert(pos2, order[1])

                    cost = route_distance(temp2, nodes)

                    if cost < best_cost:
                        best_cost = cost
                        best_route = temp2

    else:
        raise ValueError("_best_insert_sequence chỉ hỗ trợ seq dài 1 hoặc 2")

    return best_route, best_cost


def _apply_swap_1_2_once(routes, route_costs, route_loads, nodes, capacity):
    """
    Swap 1-2:
        - lấy 1 customer từ route A
        - lấy 2 customer từ route B
        - đổi cho nhau
    Có thử cả hai chiều A:1 - B:2 và A:2 - B:1.
    First improvement.
    """
    num_routes = len(routes)

    for r1 in range(num_routes):
        for r2 in range(r1 + 1, num_routes):

            # =====================================================
            # Case 1: lấy 1 khách từ r1, 2 khách từ r2
            # =====================================================
            if len(routes[r2]) >= 2:
                for i in range(len(routes[r1])):
                    single = routes[r1][i]
                    single_demand = nodes[single]["demand"]

                    for j, k in _limited_pairs(len(routes[r2])):
                        pair = [routes[r2][j], routes[r2][k]]
                        pair_demand = _sequence_demand(pair, nodes)

                        new_load_r1 = route_loads[r1] - single_demand + pair_demand
                        new_load_r2 = route_loads[r2] - pair_demand + single_demand

                        if new_load_r1 > capacity or new_load_r2 > capacity:
                            continue

                        base_r1 = _remove_indices(routes[r1], [i])
                        base_r2 = _remove_indices(routes[r2], [j, k])

                        new_r1, new_cost_r1 = _best_insert_sequence(
                            base_r1,
                            pair,
                            nodes
                        )

                        new_r2, new_cost_r2 = _best_insert_sequence(
                            base_r2,
                            [single],
                            nodes
                        )

                        old_cost = route_costs[r1] + route_costs[r2]
                        new_cost = new_cost_r1 + new_cost_r2

                        if new_cost < old_cost - EPS:
                            routes[r1] = new_r1
                            routes[r2] = new_r2

                            route_costs[r1] = new_cost_r1
                            route_costs[r2] = new_cost_r2

                            route_loads[r1] = new_load_r1
                            route_loads[r2] = new_load_r2

                            return True

            # =====================================================
            # Case 2: lấy 2 khách từ r1, 1 khách từ r2
            # =====================================================
            if len(routes[r1]) >= 2:
                for i, j in _limited_pairs(len(routes[r1])):
                    pair = [routes[r1][i], routes[r1][j]]
                    pair_demand = _sequence_demand(pair, nodes)

                    for k in range(len(routes[r2])):
                        single = routes[r2][k]
                        single_demand = nodes[single]["demand"]

                        new_load_r1 = route_loads[r1] - pair_demand + single_demand
                        new_load_r2 = route_loads[r2] - single_demand + pair_demand

                        if new_load_r1 > capacity or new_load_r2 > capacity:
                            continue

                        base_r1 = _remove_indices(routes[r1], [i, j])
                        base_r2 = _remove_indices(routes[r2], [k])

                        new_r1, new_cost_r1 = _best_insert_sequence(
                            base_r1,
                            [single],
                            nodes
                        )

                        new_r2, new_cost_r2 = _best_insert_sequence(
                            base_r2,
                            pair,
                            nodes
                        )

                        old_cost = route_costs[r1] + route_costs[r2]
                        new_cost = new_cost_r1 + new_cost_r2

                        if new_cost < old_cost - EPS:
                            routes[r1] = new_r1
                            routes[r2] = new_r2

                            route_costs[r1] = new_cost_r1
                            route_costs[r2] = new_cost_r2

                            route_loads[r1] = new_load_r1
                            route_loads[r2] = new_load_r2

                            return True

    return False


def _apply_swap_2_2_once(routes, route_costs, route_loads, nodes, capacity):
    """
    Swap 2-2:
        - lấy 2 customer từ route A
        - lấy 2 customer từ route B
        - đổi cho nhau
    First improvement.
    """
    num_routes = len(routes)

    for r1 in range(num_routes):
        if len(routes[r1]) < 2:
            continue

        for r2 in range(r1 + 1, num_routes):
            if len(routes[r2]) < 2:
                continue

            for i, j in _limited_pairs(len(routes[r1])):
                pair_1 = [routes[r1][i], routes[r1][j]]
                demand_1 = _sequence_demand(pair_1, nodes)

                for k, l in _limited_pairs(len(routes[r2])):
                    pair_2 = [routes[r2][k], routes[r2][l]]
                    demand_2 = _sequence_demand(pair_2, nodes)

                    new_load_r1 = route_loads[r1] - demand_1 + demand_2
                    new_load_r2 = route_loads[r2] - demand_2 + demand_1

                    if new_load_r1 > capacity or new_load_r2 > capacity:
                        continue

                    base_r1 = _remove_indices(routes[r1], [i, j])
                    base_r2 = _remove_indices(routes[r2], [k, l])

                    new_r1, new_cost_r1 = _best_insert_sequence(
                        base_r1,
                        pair_2,
                        nodes
                    )

                    new_r2, new_cost_r2 = _best_insert_sequence(
                        base_r2,
                        pair_1,
                        nodes
                    )

                    old_cost = route_costs[r1] + route_costs[r2]
                    new_cost = new_cost_r1 + new_cost_r2

                    if new_cost < old_cost - EPS:
                        routes[r1] = new_r1
                        routes[r2] = new_r2

                        route_costs[r1] = new_cost_r1
                        route_costs[r2] = new_cost_r2

                        route_loads[r1] = new_load_r1
                        route_loads[r2] = new_load_r2

                        return True

    return False


def multi_customer_swap(
    parent,
    route,
    fitness,
    nodes,
    capacity,
    enable_swap_1_2=True,
    enable_swap_2_2=True,
    max_rounds=1,
):
    """
    Multi-customer inter-route swap.

    Hỗ trợ:
        swap 1-2
        swap 2-2

    Không đổi số route.
    Mục tiêu:
        - giảm distance
        - cân bằng tải
        - tạo điều kiện cho route_elimination xóa route sau đó
    """
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    for _ in range(max_rounds):
        improved = False

        if enable_swap_1_2:
            improved = _apply_swap_1_2_once(
                routes,
                route_costs,
                route_loads,
                nodes,
                capacity
            )

        if not improved and enable_swap_2_2:
            improved = _apply_swap_2_2_once(
                routes,
                route_costs,
                route_loads,
                nodes,
                capacity
            )

        if not improved:
            break

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * ROUTE_PENALTY + sum(route_costs)

    return new_parent, new_route, new_fitness