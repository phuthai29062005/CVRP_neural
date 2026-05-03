from caculate import get_fitness
from Local_search.opt_2 import opt_2
from Local_search.relocation import relocation

try:
    from Local_search.route_elimination import route_elimination
    HAS_ROUTE_ELIMINATION = True
except ImportError:
    HAS_ROUTE_ELIMINATION = False

try:
    from Local_search.inter_route_swap import inter_route_swap
    HAS_INTER_ROUTE_SWAP = True
except ImportError:
    HAS_INTER_ROUTE_SWAP = False

try:
    from Local_search.two_customer_relocation import two_customer_relocation
    HAS_TWO_CUSTOMER_RELOCATION = True
except ImportError:
    HAS_TWO_CUSTOMER_RELOCATION = False


def local_search(parent, capacity, nodes, route, fitness, elite_ratio=0.15):
    """
    Local search rút gọn.
    Mỗi operator chỉ chạy 1 lần trên mỗi cá thể elite.

    Thứ tự:
        1. inter_route_swap
        2. relocation
        3. two_customer_relocation
        4. route_elimination
        5. opt_2
    """

    pop_size = len(parent)
    elite_count = max(1, int(pop_size * elite_ratio))

    fitness_sorted = sorted(fitness, key=lambda x: x[0])
    elite_indices = [idx for _, idx in fitness_sorted[:elite_count]]

    for idx in elite_indices:
        fit_val = get_fitness(parent[idx], route[idx], nodes)

        # 1. Swap giữa các route để giảm distance / cân bằng nhẹ
        if HAS_INTER_ROUTE_SWAP:
            parent[idx], route[idx], fit_val = inter_route_swap(
                parent[idx],
                route[idx],
                fit_val,
                nodes,
                capacity
            )

        # 2. Relocation 1-customer
        parent[idx], route[idx], fit_val = relocation(
            parent[idx],
            route[idx],
            fit_val,
            nodes,
            capacity
        )

        # 3. Relocation 2-customer
        if HAS_TWO_CUSTOMER_RELOCATION:
            parent[idx], route[idx], fit_val = two_customer_relocation(
                parent[idx],
                route[idx],
                fit_val,
                nodes,
                capacity
            )

        # 4. Route elimination sau khi đã relocate/swap
        if HAS_ROUTE_ELIMINATION:
            parent[idx], route[idx], fit_val = route_elimination(
                parent[idx],
                route[idx],
                fit_val,
                nodes,
                capacity
            )

        # 5. 2-opt cuối cùng để làm mượt từng route
        parent[idx], route[idx], fit_val = opt_2(
            parent[idx],
            route[idx],
            fit_val,
            nodes
        )

    # Rebuild fitness cho toàn bộ population
    new_fitness = []
    for i in range(pop_size):
        fit = get_fitness(parent[i], route[i], nodes)
        new_fitness.append((fit, i))

    new_fitness.sort()
    return parent, route, new_fitness