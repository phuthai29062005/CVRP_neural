from caculate import get_fitness
from Local_search.opt_2 import k_opt
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
    
try:
    from Local_search.multi_customer_swap import multi_customer_swap
    HAS_MULTI_CUSTOMER_SWAP = True
except ImportError:
    HAS_MULTI_CUSTOMER_SWAP = False


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

    K_OPT_VALUE = 2  # đổi thành 2 nếu muốn chạy nhanh hơn


    for idx in elite_indices:
        fit_val = get_fitness(parent[idx], route[idx], nodes)

        # 1. Chạy 1 vòng để tạo khoảng trống capacity và thử giảm route
        for _ in range(1):
            if HAS_INTER_ROUTE_SWAP:
                parent[idx], route[idx], fit_val = inter_route_swap(
                    parent[idx],
                    route[idx],
                    fit_val,
                    nodes,
                    capacity
                )

            if HAS_MULTI_CUSTOMER_SWAP:
                parent[idx], route[idx], fit_val = multi_customer_swap(
                    parent[idx],
                    route[idx],
                    fit_val,
                    nodes,
                    capacity,
                    enable_swap_1_2=True,
                    enable_swap_2_2=True,
                    max_rounds=1,
                )
            parent[idx], route[idx], fit_val = relocation(
                parent[idx],
                route[idx],
                fit_val,
                nodes,
                capacity
            )

            if HAS_TWO_CUSTOMER_RELOCATION:
                parent[idx], route[idx], fit_val = two_customer_relocation(
                    parent[idx],
                    route[idx],
                    fit_val,
                    nodes,
                    capacity
                )

            if HAS_ROUTE_ELIMINATION:
                parent[idx], route[idx], fit_val = route_elimination(
                    parent[idx],
                    route[idx],
                    fit_val,
                    nodes,
                    capacity
                )

        # 2. Sau khi cấu trúc route ổn hơn, mới chạy k-opt một lần
        parent[idx], route[idx], fit_val = k_opt(
            parent[idx],
            route[idx],
            fit_val,
            nodes,
            k=K_OPT_VALUE,
            max_route_len_for_3opt=40,
            three_opt_rounds=1,
        )

        # 3. Sau k-opt, thử xóa route thêm lần nữa
        if HAS_ROUTE_ELIMINATION:
            parent[idx], route[idx], fit_val = route_elimination(
                parent[idx],
                route[idx],
                fit_val,
                nodes,
                capacity
            )

    # Rebuild fitness cho toàn bộ population
    new_fitness = []
    for i in range(pop_size):
        fit = get_fitness(parent[i], route[i], nodes)
        new_fitness.append((fit, i))

    new_fitness.sort()
    return parent, route, new_fitness