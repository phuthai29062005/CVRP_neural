
import os
import random
import math
import time

from read_data import read_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FILE_PATH = os.path.join(
    BASE_DIR, "ML4VRP2026", "Instances", "cvrp", "vrp", "X-n401-k29.vrp"
)

# =========================================================
# FE BUDGET NOTE
# ---------------------------------------------------------
# Main algorithm of user: ~11,500 FEs
# This NSGA-II + LS uses:
#   pop_size = 100
#   generations = 107
#   LS every 5 generations on top 15% individuals
# Approx FE count:
#   init = 100
#   offspring evals = 107 * 100 = 10,700
#   LS evals ≈ floor(107/5) * 15 * 2 = 630
#   total ≈ 11,430
# =========================================================

POP_SIZE = 100
GENERATIONS = 150
LS_EVERY = 5
LS_ELITE_RATIO = 0.15
SEED = 42


def get_node_ids(nodes):
    if isinstance(nodes, dict):
        return sorted(nodes.keys())
    return list(range(len(nodes)))


def get_node_data(nodes, node_id):
    if isinstance(nodes, dict):
        return nodes[node_id]
    return nodes[node_id]


def get_xy_and_demand(node):
    if isinstance(node, dict):
        if "x" in node and "y" in node:
            x = node["x"]
            y = node["y"]
        elif "coord" in node:
            x, y = node["coord"]
        elif "coords" in node:
            x, y = node["coords"]
        else:
            raise ValueError(f"Cannot extract coordinates from node dict: {node}")

        if "demand" in node:
            demand = node["demand"]
        elif "dem" in node:
            demand = node["dem"]
        else:
            demand = 0

        return float(x), float(y), float(demand)

    if isinstance(node, (list, tuple)):
        if len(node) == 4:
            return float(node[1]), float(node[2]), float(node[3])
        elif len(node) == 3:
            return float(node[0]), float(node[1]), float(node[2])

    raise ValueError(f"Unsupported node format: {node}")


def infer_depot_id(nodes):
    node_ids = get_node_ids(nodes)
    if 1 in node_ids:
        return 1
    return node_ids[0]


def build_dist_matrix(nodes):
    node_ids = get_node_ids(nodes)
    dist = {}

    for i in node_ids:
        xi, yi, _ = get_xy_and_demand(get_node_data(nodes, i))
        dist[i] = {}
        for j in node_ids:
            xj, yj, _ = get_xy_and_demand(get_node_data(nodes, j))
            dist[i][j] = math.hypot(xi - xj, yi - yj)

    return dist


def decode(perm, capacity, nodes, dist, depot_id):
    routes = []
    route = []
    load = 0.0

    for c in perm:
        _, _, demand = get_xy_and_demand(get_node_data(nodes, c))

        if load + demand <= capacity:
            route.append(c)
            load += demand
        else:
            if route:
                routes.append(route)
            route = [c]
            load = demand

    if route:
        routes.append(route)

    total_dist = 0.0
    for r in routes:
        prev = depot_id
        for c in r:
            total_dist += dist[prev][c]
            prev = c
        total_dist += dist[prev][depot_id]

    return routes, len(routes), total_dist


def evaluate_perm(perm, capacity, nodes, dist, depot_id):
    routes, r, d = decode(perm, capacity, nodes, dist, depot_id)
    return {
        "perm": perm[:],
        "routes": routes,
        "r": r,
        "d": d,
        "rank": None,
        "cd": 0.0,
    }


def route_demand(route, nodes):
    total = 0.0
    for c in route:
        _, _, dem = get_xy_and_demand(get_node_data(nodes, c))
        total += dem
    return total


def route_distance(route, dist, depot_id):
    if not route:
        return 0.0
    total = dist[depot_id][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][depot_id]
    return total


def flatten_routes(routes):
    perm = []
    for r in routes:
        perm.extend(r)
    return perm


def init_pop(nodes, pop_size, depot_id):
    customer_ids = [nid for nid in get_node_ids(nodes) if nid != depot_id]
    pop = []

    for _ in range(pop_size):
        arr = customer_ids.copy()
        random.shuffle(arr)
        pop.append(arr)

    return pop


def ox(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = p1[a:b]

    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child


def swap_mutation(p):
    p = p.copy()
    i, j = random.sample(range(len(p)), 2)
    p[i], p[j] = p[j], p[i]
    return p


def inversion_mutation(p):
    p = p.copy()
    i, j = sorted(random.sample(range(len(p)), 2))
    p[i:j+1] = reversed(p[i:j+1])
    return p


def mutate(p, swap_rate=0.25, inversion_rate=0.20):
    child = p[:]
    if random.random() < swap_rate:
        child = swap_mutation(child)
    if random.random() < inversion_rate:
        child = inversion_mutation(child)
    return child


def dominates(a, b):
    return (
        a["r"] <= b["r"] and a["d"] <= b["d"]
        and
        (a["r"] < b["r"] or a["d"] < b["d"])
    )


def fast_sort(pop):
    fronts = [[]]

    for p in pop:
        p["dom"] = []
        p["cnt"] = 0

    for i, p in enumerate(pop):
        for j, q in enumerate(pop):
            if i == j:
                continue
            if dominates(p, q):
                p["dom"].append(j)
            elif dominates(q, p):
                p["cnt"] += 1

        if p["cnt"] == 0:
            p["rank"] = 0
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        nxt = []
        for p_idx in fronts[i]:
            for q_idx in pop[p_idx]["dom"]:
                pop[q_idx]["cnt"] -= 1
                if pop[q_idx]["cnt"] == 0:
                    pop[q_idx]["rank"] = i + 1
                    nxt.append(q_idx)
        i += 1
        fronts.append(nxt)

    if not fronts[-1]:
        fronts.pop()

    return fronts


def crowding(pop, front):
    if not front:
        return

    for idx in front:
        pop[idx]["cd"] = 0.0

    if len(front) == 1:
        pop[front[0]]["cd"] = float("inf")
        return

    if len(front) == 2:
        pop[front[0]]["cd"] = float("inf")
        pop[front[1]]["cd"] = float("inf")
        return

    for key in ["r", "d"]:
        front_sorted = sorted(front, key=lambda idx: pop[idx][key])
        pop[front_sorted[0]]["cd"] = float("inf")
        pop[front_sorted[-1]]["cd"] = float("inf")

        mn = pop[front_sorted[0]][key]
        mx = pop[front_sorted[-1]][key]
        if abs(mx - mn) < 1e-12:
            continue

        for i in range(1, len(front_sorted) - 1):
            prev_val = pop[front_sorted[i - 1]][key]
            next_val = pop[front_sorted[i + 1]][key]
            if pop[front_sorted[i]]["cd"] != float("inf"):
                pop[front_sorted[i]]["cd"] += (next_val - prev_val) / (mx - mn)


def select(pop, size):
    fronts = fast_sort(pop)
    new = []

    for front in fronts:
        crowding(pop, front)
        sorted_front = sorted(front, key=lambda idx: (pop[idx]["rank"], -pop[idx]["cd"]))

        if len(new) + len(sorted_front) <= size:
            new.extend(pop[idx] for idx in sorted_front)
        else:
            remain = size - len(new)
            new.extend(pop[idx] for idx in sorted_front[:remain])
            break

    return new


def tournament(pop):
    a, b = random.sample(pop, 2)
    if a["rank"] < b["rank"]:
        return a
    if a["rank"] > b["rank"]:
        return b
    return a if a["cd"] > b["cd"] else b


def intra_route_2opt(route, dist, depot_id):
    if len(route) < 3:
        return route[:], route_distance(route, dist, depot_id)

    best_route = route[:]
    best_cost = route_distance(best_route, dist, depot_id)
    improved = True

    while improved:
        improved = False
        n = len(best_route)
        for i in range(n - 1):
            a = depot_id if i == 0 else best_route[i - 1]
            b = best_route[i]
            for j in range(i + 1, n):
                c = best_route[j]
                d = depot_id if j == n - 1 else best_route[j + 1]
                delta = -dist[a][b] - dist[c][d] + dist[a][c] + dist[b][d]
                if delta < -1e-12:
                    best_route[i:j + 1] = reversed(best_route[i:j + 1])
                    best_cost += delta
                    improved = True
                    break
            if improved:
                break

    return best_route, best_cost


def same_route_reloc_delta(route, i, j, dist, depot_id):
    old_cost = route_distance(route, dist, depot_id)
    node = route[i]
    temp = route[:i] + route[i + 1:]
    temp = temp[:j] + [node] + temp[j:]
    new_cost = route_distance(temp, dist, depot_id)
    return temp, new_cost - old_cost, new_cost


def inter_route_reloc_delta(route1, i, route2, j, dist, depot_id):
    b = route1[i]
    a = depot_id if i == 0 else route1[i - 1]
    c = depot_id if i == len(route1) - 1 else route1[i + 1]

    u = depot_id if j == 0 else route2[j - 1]
    v = depot_id if j == len(route2) else route2[j]

    delta_remove = -dist[a][b] - dist[b][c] + dist[a][c]
    delta_insert = -dist[u][v] + dist[u][b] + dist[b][v]

    return delta_remove + delta_insert


def relocation_ls(ind, capacity, nodes, dist, depot_id):
    routes = [r[:] for r in ind["routes"]]
    route_costs = [route_distance(r, dist, depot_id) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True
    while improved:
        improved = False

        for r1 in range(len(routes)):
            for i in range(len(routes[r1])):
                node = routes[r1][i]
                _, _, demand = get_xy_and_demand(get_node_data(nodes, node))

                for r2 in range(len(routes)):
                    if r1 == r2:
                        for j in range(len(routes[r1])):
                            if j == i:
                                continue
                            new_route_r, delta, new_cost = same_route_reloc_delta(
                                routes[r1], i, j, dist, depot_id
                            )
                            if delta < -1e-12:
                                routes[r1] = new_route_r
                                route_costs[r1] = new_cost
                                improved = True
                                break
                        if improved:
                            break
                    else:
                        if route_loads[r2] + demand > capacity:
                            continue

                        for j in range(len(routes[r2]) + 1):
                            delta = inter_route_reloc_delta(routes[r1], i, routes[r2], j, dist, depot_id)

                            remove_empty_source = (len(routes[r1]) == 1)
                            penalty_delta = -1000.0 if remove_empty_source else 0.0
                            total_delta = delta + penalty_delta

                            if total_delta < -1e-12:
                                node_to_move = routes[r1][i]
                                new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                                new_r2 = routes[r2][:j] + [node_to_move] + routes[r2][j:]

                                routes[r2] = new_r2
                                route_costs[r2] = route_distance(new_r2, dist, depot_id)
                                route_loads[r2] += demand

                                if remove_empty_source:
                                    del routes[r1]
                                    del route_costs[r1]
                                    del route_loads[r1]
                                else:
                                    routes[r1] = new_r1
                                    route_costs[r1] = route_distance(new_r1, dist, depot_id)
                                    route_loads[r1] -= demand

                                improved = True
                                break
                        if improved:
                            break

                if improved:
                    break
            if improved:
                break

    new_perm = flatten_routes(routes)
    return {
        "perm": new_perm,
        "routes": routes,
        "r": len(routes),
        "d": sum(route_costs),
        "rank": None,
        "cd": 0.0,
    }


def two_opt_ls(ind, dist, depot_id):
    routes = []
    total_d = 0.0

    for r in ind["routes"]:
        new_r, new_cost = intra_route_2opt(r, dist, depot_id)
        routes.append(new_r)
        total_d += new_cost

    new_perm = flatten_routes(routes)
    return {
        "perm": new_perm,
        "routes": routes,
        "r": len(routes),
        "d": total_d,
        "rank": None,
        "cd": 0.0,
    }


def local_search_population(pop, capacity, nodes, dist, depot_id, elite_ratio=0.15):
    pop_size = len(pop)
    elite_count = max(1, int(pop_size * elite_ratio))
    order = sorted(range(pop_size), key=lambda i: 1000.0 * pop[i]["r"] + pop[i]["d"])

    ls_evals = 0
    for k in range(elite_count):
        idx = order[k]
        improved = relocation_ls(pop[idx], capacity, nodes, dist, depot_id)
        ls_evals += 1
        improved = two_opt_ls(improved, dist, depot_id)
        ls_evals += 1
        pop[idx] = improved

    return pop, ls_evals


def run_nsga_ls(
    file_path=DEFAULT_FILE_PATH,
    pop_size=POP_SIZE,
    gen=GENERATIONS,
    seed=SEED,
    ls_every=LS_EVERY,
    ls_elite_ratio=LS_ELITE_RATIO,
):
    random.seed(seed)

    _, capacity, nodes = read_data(file_path)
    depot_id = infer_depot_id(nodes)
    dist = build_dist_matrix(nodes)

    perms = init_pop(nodes, pop_size, depot_id)
    pop = [evaluate_perm(p, capacity, nodes, dist, depot_id) for p in perms]
    total_fes = pop_size

    pop = select(pop, pop_size)

    history = []
    start = time.time()

    for g in range(1, gen + 1):
        offspring = []

        while len(offspring) < pop_size:
            p1 = tournament(pop)
            p2 = tournament(pop)
            child_perm = mutate(ox(p1["perm"], p2["perm"]))
            child = evaluate_perm(child_perm, capacity, nodes, dist, depot_id)
            offspring.append(child)

        total_fes += pop_size
        pop = select(pop + offspring, pop_size)

        if g % ls_every == 0:
            pop, ls_evals = local_search_population(
                pop, capacity, nodes, dist, depot_id, elite_ratio=ls_elite_ratio
            )
            total_fes += ls_evals
            pop = select(pop, pop_size)

        best = min(pop, key=lambda x: 1000.0 * x["r"] + x["d"])
        best_f = 1000.0 * best["r"] + best["d"]
        history.append(best_f)

        print(
            f"Gen {g:3d}: F = {best_f:.4f} | "
            f"routes = {best['r']} | distance = {best['d']:.4f} | total_FEs ~= {total_fes}"
        )

    best = min(pop, key=lambda x: 1000.0 * x["r"] + x["d"])
    elapsed = time.time() - start

    print(f"\\nDone in {elapsed:.2f}s")
    print(f"Approx total FEs = {total_fes}")

    return best, history, total_fes


def save(instance_path, best, history, suffix="_nsga_ls"):
    out_dir = os.path.join(BASE_DIR, "Self_Solutions")
    os.makedirs(out_dir, exist_ok=True)

    name = os.path.basename(instance_path).split(".")[0]

    fitness_file = os.path.join(out_dir, f"{name}_fitness{suffix}.txt")
    with open(fitness_file, "w") as f:
        for i, val in enumerate(history, 1):
            f.write(f"{i}\\t{val}\\n")

    route_file = os.path.join(out_dir, f"{name}_routes{suffix}.txt")
    with open(route_file, "w") as f:
        for i, r in enumerate(best["routes"], 1):
            f.write(f"Route #{i}: {' '.join(map(str, r))}\\n")

    print("Saved fitness to:", fitness_file)
    print("Saved routes to:", route_file)


if __name__ == "__main__":
    best, hist, total_fes = run_nsga_ls(
        file_path=DEFAULT_FILE_PATH,
        pop_size=POP_SIZE,
        gen=GENERATIONS,
        seed=SEED,
        ls_every=LS_EVERY,
        ls_elite_ratio=LS_ELITE_RATIO,
    )
    save(DEFAULT_FILE_PATH, best, hist, suffix="_nsga_ls")
