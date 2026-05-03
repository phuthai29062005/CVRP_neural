import os
import random
import numpy as np

from read_data import read_data
from GA import GA
from caculate import get_route, get_fitness
from Local_search.local_search import local_search
from Local_search.local_search_utils import (
    build_distance_matrix,
    build_k_nearest_neighbors,
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INSTANCE_PATH = os.path.join(
    BASE_DIR,
    "ML4VRP2026",
    "Instances",
    "cvrp",
    "vrp",
    "X-n101-k25.vrp"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "New_Solutions")

GENERATION = 50
POPULATION = 100
ELITE_RATIO = 0.10
LOCAL_SEARCH_EVERY = 5
LOCAL_SEARCH_ELITE_RATIO = 0.20

RENEW_PATIENCE = 8
RENEW_AFTER_GEN = 8


USE_NEURAL_FILL = True
NEURAL_DECODE_TYPE = "sampling"

# Ưu tiên checkpoint train_v2 mới.
# Nếu đường dẫn của bạn khác, sửa ở đây.
NEURAL_CKPT_CANDIDATES = [
    os.path.join(
        BASE_DIR,
        "Train_Neural",
        "checkpoints_neural_fill_v2",
        "model_best_sampling.pt"
    ),
    os.path.join(
        BASE_DIR,
        "checkpoints_neural_fill_v2",
        "model_best_sampling.pt"
    ),
]


# =========================================================
# HELPERS
# =========================================================

def resolve_neural_checkpoint():
    """
    Tìm checkpoint neural hợp lệ.
    """
    if not USE_NEURAL_FILL:
        return None

    for ckpt_path in NEURAL_CKPT_CANDIDATES:
        if os.path.exists(ckpt_path):
            return ckpt_path

    raise FileNotFoundError(
        "No neural checkpoint found. Checked:\n"
        + "\n".join(NEURAL_CKPT_CANDIDATES)
    )


def clone_list(x):
    """
    Copy list an toàn.
    """
    return x.copy() if hasattr(x, "copy") else list(x)


def get_pop(population, dimension):
    """
    Tạo population ban đầu.
    Customer id chạy từ 2 đến dimension.
    Depot là node 1.
    """
    parent = []
    customers = list(range(2, dimension + 1))

    for _ in range(population):
        random.shuffle(customers)
        parent.append(customers.copy())

    return parent


def evaluate_population(parent, route, nodes, dist_matrix):
    fitness = []

    for i in range(len(parent)):
        fit = get_fitness(parent[i], route[i], nodes, dist_matrix)
        fitness.append((fit, i))

    fitness.sort()
    return fitness


def update_global_best(fitness, parent, route, best_fit, best_parent, best_route):
    """
    Cập nhật nghiệm tốt nhất toàn cục.
    """
    fitness.sort()

    current_best_fit, current_best_idx = fitness[0]

    if current_best_fit < best_fit:
        best_fit = current_best_fit
        best_parent = clone_list(parent[current_best_idx])
        best_route = clone_list(route[current_best_idx])
        improved = True
    else:
        improved = False

    return best_fit, best_parent, best_route, improved


def build_index_mapping(fitness):
    """
    Mapping từ old_idx trong parent hiện tại sang vị trí của nó trong fitness đã sort.
    """
    index_mapping = {}

    for sorted_pos, (_, old_idx) in enumerate(fitness):
        index_mapping[old_idx] = sorted_pos

    return index_mapping


def should_renew(stale_count, gen):
    """
    Renew khi global best không cải thiện sau một số generation.
    """
    return (
        stale_count >= RENEW_PATIENCE
        and gen > RENEW_AFTER_GEN
    )


def renew_population(population, dimension, capacity, nodes, dist_matrix):
    parent = get_pop(population, dimension)
    route = get_route(parent, dimension, population, capacity, nodes)
    fitness = evaluate_population(parent, route, nodes, dist_matrix)

    return parent, route, fitness


def add_individual(new_parent, new_route, new_fitness, individual, individual_route, individual_fit):
    """
    Thêm một cá thể vào population mới.
    """
    new_idx = len(new_parent)

    new_parent.append(clone_list(individual))
    new_route.append(clone_list(individual_route))
    new_fitness.append((individual_fit, new_idx))


def extract_route_list(best_parent, best_route):
    """
    Chuyển best_parent + marker best_route thành list các route.
    marker == 1 nghĩa là bắt đầu route mới.
    """
    route_list = []
    current_route = []

    for j, marker in enumerate(best_route):
        if marker == 1 and current_route:
            route_list.append(current_route.copy())
            current_route = [best_parent[j]]
        else:
            current_route.append(best_parent[j])

    if current_route:
        route_list.append(current_route)

    return route_list


def save_fitness_history(output_dir, instance_name, fitness_history):
    """
    Lưu lịch sử global best fitness.
    """
    os.makedirs(output_dir, exist_ok=True)

    fitness_file = os.path.join(
        output_dir,
        f"{instance_name}_fitness.txt"
    )

    with open(fitness_file, "w") as f:
        for gen_num, fit_val in enumerate(fitness_history, start=1):
            f.write(f"{gen_num}\t{fit_val}\n")

    print(f"Saved fitness history to {fitness_file}")


def save_best_routes(output_dir, instance_name, route_list):
    """
    Lưu best routes.
    """
    os.makedirs(output_dir, exist_ok=True)

    route_file = os.path.join(
        output_dir,
        f"{instance_name}_routes.txt"
    )

    with open(route_file, "w") as f:
        for route_num, route_customers in enumerate(route_list, start=1):
            f.write(
                f"Route #{route_num}: "
                f"{' '.join(map(str, route_customers))}\n"
            )

    print(f"Saved best routes to {route_file}")


# =========================================================
# MAIN
# =========================================================

def main():
    dimension, capacity, nodes = read_data(INSTANCE_PATH)
    
    dist_matrix = build_distance_matrix(nodes, dimension)
    nearest_neighbors = build_k_nearest_neighbors(
        dist_matrix,
        dimension,
        k=10,
    )
    
    instance_name = os.path.basename(INSTANCE_PATH).split(".")[0]

    print("=" * 70)
    print(f"Instance: {instance_name}")
    print(f"Dimension: {dimension}")
    print(f"Capacity: {capacity}")
    print(f"Generation: {GENERATION}")
    print(f"Population: {POPULATION}")
    print("=" * 70)

    neural_ckpt_path = resolve_neural_checkpoint()

    if USE_NEURAL_FILL:
        print(f"Using neural checkpoint: {neural_ckpt_path}")
        print(f"Neural decode type: {NEURAL_DECODE_TYPE}")

    # -----------------------------
    # Initial population
    # -----------------------------
    parent = get_pop(POPULATION, dimension)
    route = get_route(parent, dimension, POPULATION, capacity, nodes)
    fitness = evaluate_population(parent, route, nodes, dist_matrix)

    best_fit = np.inf
    best_parent = None
    best_route = None

    phase_best_fit = np.inf
    
    best_fitness_history = []
    renew_count = 0
    stale_count = 0

    # -----------------------------
    # Evolution loop
    # -----------------------------
    for gen in range(1, GENERATION + 1):
        # Update global best: lưu nghiệm tốt nhất toàn bộ quá trình
        best_fit, best_parent, best_route, improved_global = update_global_best(
            fitness, parent, route, best_fit, best_parent, best_route
        )

        # Update phase best: dùng để quyết định renew
        fitness.sort()
        current_phase_best = fitness[0][0]

        if current_phase_best < phase_best_fit:
            phase_best_fit = current_phase_best
            stale_count = 0
        else:
            stale_count += 1

        # Renew if stagnated
        if should_renew(stale_count, gen):
            renew_count += 1
            stale_count = 0

            parent, route, fitness = renew_population(
                POPULATION,
                dimension,
                capacity,
                nodes,
                dist_matrix,
            )

            # Reset phase sau khi renew
            fitness.sort()
            phase_best_fit = fitness[0][0]

        fitness.sort()
        current_best = fitness[0][0]

        index_mapping = build_index_mapping(fitness)

        new_parent = []
        new_route = []
        new_fitness = []

        # -----------------------------
        # Elite selection
        # -----------------------------
        elite_count = int(POPULATION * ELITE_RATIO)

        for j in range(elite_count):
            old_idx = fitness[j][1]

            add_individual(
                new_parent,
                new_route,
                new_fitness,
                parent[old_idx],
                route[old_idx],
                fitness[j][0]
            )

        # -----------------------------
        # Crossover / neural fill
        # -----------------------------
        for idx in range(POPULATION - elite_count):
            par1 = fitness[idx][1]
            par2 = fitness[(idx + 1) % POPULATION][1]

            par3 = np.random.randint(len(parent))
            while par3 == par1 or par3 == par2:
                par3 = np.random.randint(len(parent))

            child, child_route, child_fitness = GA(
                parent,
                route,
                par1,
                par2,
                par3,
                dimension=dimension,
                capacity=capacity,
                nodes=nodes,
                dist_matrix=dist_matrix,
                nearest_neighbors=nearest_neighbors,
                use_neural_fill=USE_NEURAL_FILL,
                neural_ckpt_path=neural_ckpt_path,
                neural_decode_type=NEURAL_DECODE_TYPE,
            )

            par1_fitness = fitness[index_mapping[par1]][0]
            par2_fitness = fitness[index_mapping[par2]][0]
            par3_fitness = fitness[index_mapping[par3]][0]

            best_parent_fitness = min(
                par1_fitness,
                par2_fitness,
                par3_fitness
            )

            # Chỉ nhận child nếu tốt hơn cả 3 parent.
            if child_fitness < best_parent_fitness:
                add_individual(
                    new_parent,
                    new_route,
                    new_fitness,
                    child,
                    child_route,
                    child_fitness
                )
            else:
                # Nếu child không tốt hơn, giữ parent tốt nhất trong 3 parent.
                if par1_fitness <= par2_fitness and par1_fitness <= par3_fitness:
                    selected_idx = par1
                    selected_fit = par1_fitness
                elif par2_fitness <= par3_fitness:
                    selected_idx = par2
                    selected_fit = par2_fitness
                else:
                    selected_idx = par3
                    selected_fit = par3_fitness

                add_individual(
                    new_parent,
                    new_route,
                    new_fitness,
                    parent[selected_idx],
                    route[selected_idx],
                    selected_fit
                )

        parent = new_parent
        route = new_route
        fitness = new_fitness

        # -----------------------------
        # Local search
        # -----------------------------
        if gen % LOCAL_SEARCH_EVERY == 0:
            parent, route, fitness = local_search(
                parent,
                capacity,
                nodes,
                route,
                fitness,
                elite_ratio=LOCAL_SEARCH_ELITE_RATIO,
                dist_matrix=dist_matrix,
                nearest_neighbors=nearest_neighbors,
            )

        fitness.sort()

        best_fit, best_parent, best_route, improved_after_global = update_global_best(
            fitness, parent, route, best_fit, best_parent, best_route
        )

        fitness.sort()
        current_phase_best = fitness[0][0]

        if current_phase_best < phase_best_fit:
            phase_best_fit = current_phase_best
            stale_count = 0

        best_fitness_history.append(best_fit)

        fitness.sort()
        current_best = fitness[0][0]
        current_best_idx = fitness[0][1]
        current_route_count = sum(route[current_best_idx])

        global_route_count = sum(best_route) if best_route is not None else 0

        print(
            f"Gen {gen:03d}/{GENERATION} | "
            f"Current best = {current_best:.4f} | "
            f"Current routes = {current_route_count} | "
            f"Phase best = {phase_best_fit:.4f} | "
            f"Global best = {best_fit:.4f} | "
            f"Global routes = {global_route_count} | "
            f"Stale = {stale_count} | "
            f"Renew = {renew_count}"
        )

    # -----------------------------
    # Save results
    # -----------------------------
    route_list = extract_route_list(best_parent, best_route)

    save_fitness_history(
        OUTPUT_DIR,
        instance_name,
        best_fitness_history
    )

    save_best_routes(
        OUTPUT_DIR,
        instance_name,
        route_list
    )

    print("=" * 70)
    print(f"Best fitness: {best_fit}")
    print(f"Number of routes: {len(route_list)}")
    print("=" * 70)


if __name__ == "__main__":
    main()