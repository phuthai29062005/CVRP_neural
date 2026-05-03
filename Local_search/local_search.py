from caculate import separate_routes
from Local_search.local_search_utils import DEPOT_ID, ROUTE_PENALTY
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


def _fitness(parent, route, nodes, dist_matrix=None):
    """Fitness local: 1000 * routes + total_distance."""
    routes = separate_routes(parent, route)
    total_distance = sum(_route_distance(r, nodes, dist_matrix) for r in routes)

    return ROUTE_PENALTY * len(routes) + total_distance


def _run_operator(
    op_name,
    parent_i,
    route_i,
    fit_val,
    nodes,
    capacity,
    dist_matrix,
    nearest_neighbors,
    generation,
    renew_count,
):
    """Wrapper để gọi operator với tham số phù hợp."""
    if op_name == "inter_route_swap":
        return inter_route_swap(
            parent_i,
            route_i,
            fit_val,
            nodes,
            capacity,
            dist_matrix=dist_matrix,
            nearest_neighbors=nearest_neighbors,
        )

    if op_name == "relocation":
        return relocation(
            parent_i,
            route_i,
            fit_val,
            nodes,
            capacity,
            dist_matrix=dist_matrix,
            nearest_neighbors=nearest_neighbors,
        )

    if op_name == "two_customer_relocation":
        return two_customer_relocation(
            parent_i,
            route_i,
            fit_val,
            nodes,
            capacity,
            dist_matrix=dist_matrix,
            nearest_neighbors=nearest_neighbors,
        )

    if op_name == "route_elimination":
        return route_elimination(
            parent_i,
            route_i,
            fit_val,
            nodes,
            capacity,
            dist_matrix=dist_matrix,
            nearest_neighbors=nearest_neighbors,
            generation=generation,
            renew_count=renew_count,
        )

    if op_name == "opt_2":
        return opt_2(
            parent_i,
            route_i,
            fit_val,
            nodes,
            dist_matrix=dist_matrix,
            nearest_neighbors=nearest_neighbors,
        )

    raise ValueError(f"Unknown local search operator: {op_name}")


def local_search(
    parent,
    capacity,
    nodes,
    route,
    fitness,
    elite_ratio=0.15,
    dist_matrix=None,
    nearest_neighbors=None,
    generation=None,
    renew_count=0,
    max_vnd_rounds=30,
):
    """
    VND Local Search.

    Thay đổi so với bản cũ:
    - Không còn chạy tuần tự 1 chiều đúng 1 lần.
    - Dùng VND loop:
        + chạy operator theo thứ tự
        + nếu operator improve thì restart lại từ operator đầu
        + dừng khi không operator nào improve nữa
    - Có max_vnd_rounds để tránh vòng lặp quá dài.

    Thứ tự operator:
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

    operators = []

    if HAS_INTER_ROUTE_SWAP:
        operators.append("inter_route_swap")

    operators.append("relocation")

    if HAS_TWO_CUSTOMER_RELOCATION:
        operators.append("two_customer_relocation")

    if HAS_ROUTE_ELIMINATION:
        operators.append("route_elimination")

    operators.append("opt_2")

    for idx in elite_indices:
        fit_val = _fitness(
            parent[idx],
            route[idx],
            nodes,
            dist_matrix=dist_matrix,
        )

        op_idx = 0
        vnd_rounds = 0

        while op_idx < len(operators) and vnd_rounds < max_vnd_rounds:
            op_name = operators[op_idx]

            old_fit = fit_val

            new_parent_i, new_route_i, new_fit = _run_operator(
                op_name=op_name,
                parent_i=parent[idx],
                route_i=route[idx],
                fit_val=fit_val,
                nodes=nodes,
                capacity=capacity,
                dist_matrix=dist_matrix,
                nearest_neighbors=nearest_neighbors,
                generation=generation,
                renew_count=renew_count,
            )

            if new_fit < old_fit - EPS:
                parent[idx] = new_parent_i
                route[idx] = new_route_i
                fit_val = new_fit

                # VND: có improvement thì restart từ operator đầu
                op_idx = 0
                vnd_rounds += 1
            else:
                # Không improve thì chuyển sang neighborhood tiếp theo
                op_idx += 1

    # Rebuild fitness cho toàn bộ population
    new_fitness = []

    for i in range(pop_size):
        fit = _fitness(
            parent[i],
            route[i],
            nodes,
            dist_matrix=dist_matrix,
        )
        new_fitness.append((fit, i))

    new_fitness.sort()

    return parent, route, new_fitness
