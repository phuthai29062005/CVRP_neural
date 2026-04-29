from Local_search.opt_2 import opt_2
from Local_search.relocation import relocation


def local_search(parent, capacity, nodes, route, fitness, elite_ratio=0.15):
    """
    Chỉ chạy local search cho top elite_ratio cá thể tốt nhất.
    fitness: list[(fit_val, idx)]
    """
    pop_size = len(parent)
    elite_count = max(1, int(pop_size * elite_ratio))

    # lấy thứ tự theo fitness hiện tại
    order = sorted(range(pop_size), key=lambda i: fitness[i][0])

    for k in range(elite_count):
        i = order[k]
        fit_val, idx = fitness[i]

        # relocation trước
        parent[i], route[i], fit_val = relocation(
            parent[i], route[i], fit_val, nodes, capacity
        )

        # rồi 2-opt refine
        parent[i], route[i], fit_val = opt_2(
            parent[i], route[i], fit_val, nodes
        )

        fitness[i] = (fit_val, idx)

    return parent, route, fitness