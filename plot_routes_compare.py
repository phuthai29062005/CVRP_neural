import os
import re
import math
import argparse
import matplotlib.pyplot as plt

DEPOT_ID = 1


# =========================================================
# READ VRP INSTANCE
# =========================================================
def read_vrp(file_path):
    dimension = None
    capacity = None
    coords = {}
    demands = {}

    current_section = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line == "EOF":
                continue

            if line.startswith("NODE_COORD_SECTION"):
                current_section = "NODE_COORD_SECTION"
                continue
            elif line.startswith("DEMAND_SECTION"):
                current_section = "DEMAND_SECTION"
                continue
            elif line.startswith("DEPOT_SECTION"):
                current_section = "DEPOT_SECTION"
                continue

            if current_section is None:
                if ":" in line:
                    key, value = [x.strip() for x in line.split(":", 1)]
                    if key == "DIMENSION":
                        dimension = int(value)
                    elif key == "CAPACITY":
                        capacity = int(value)

            elif current_section == "NODE_COORD_SECTION":
                parts = line.split()
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[node_id] = (x, y)

            elif current_section == "DEMAND_SECTION":
                parts = line.split()
                node_id = int(parts[0])
                demand = int(parts[1])
                demands[node_id] = demand

    if dimension is None:
        raise ValueError("Không đọc được DIMENSION từ file .vrp")

    return dimension, capacity, coords, demands


# =========================================================
# READ ROUTE FILE
# Format hỗ trợ:
# Route #1: 2 17 35 44
# Route #2: 9 10 81 72
# =========================================================
def read_routes(route_file):
    routes = []

    with open(route_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if ":" in line:
                line = line.split(":", 1)[1]

            nums = [int(x) for x in re.findall(r"-?\d+", line)]
            nums = [x for x in nums if x != DEPOT_ID and x != -1]

            if nums:
                routes.append(nums)

    if not routes:
        raise ValueError(f"Không đọc được route nào từ file: {route_file}")

    return routes


# =========================================================
# DISTANCE / CHECK
# =========================================================
def euclid(coords, a, b):
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.hypot(ax - bx, ay - by)


def route_distance(route, coords):
    if not route:
        return 0.0

    total = euclid(coords, DEPOT_ID, route[0])
    for i in range(len(route) - 1):
        total += euclid(coords, route[i], route[i + 1])
    total += euclid(coords, route[-1], DEPOT_ID)
    return total


def solution_distance(routes, coords):
    return sum(route_distance(r, coords) for r in routes)


def route_load(route, demands):
    return sum(demands.get(c, 0) for c in route)


def check_solution(routes, dimension):
    all_customers = []
    for r in routes:
        all_customers.extend(r)

    expected = set(range(2, dimension + 1))
    actual = set(all_customers)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    duplicates = []
    seen = set()
    for x in all_customers:
        if x in seen and x not in duplicates:
            duplicates.append(x)
        seen.add(x)

    return missing, duplicates, extra


# =========================================================
# PLOT
# =========================================================
def plot_routes(instance_path, route_path, save_path=None, show_node_id=False, show_route_id=False):
    dimension, capacity, coords, demands = read_vrp(instance_path)
    routes = read_routes(route_path)

    missing, duplicates, extra = check_solution(routes, dimension)

    print("=" * 70)
    print(f"Instance: {os.path.basename(instance_path)}")
    print(f"Route file: {route_path}")
    print(f"Number of routes: {len(routes)}")
    print(f"Total distance: {solution_distance(routes, coords):.4f}")
    print(f"Missing customers: {missing}")
    print(f"Duplicate customers: {duplicates}")
    print(f"Extra customers: {extra}")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 8))

    # vẽ customer
    customer_x = []
    customer_y = []
    for node_id, (x, y) in coords.items():
        if node_id == DEPOT_ID:
            continue
        customer_x.append(x)
        customer_y.append(y)

    ax.scatter(customer_x, customer_y, s=18, alpha=0.45)

    # vẽ depot
    depot_x, depot_y = coords[DEPOT_ID]
    ax.scatter(
        [depot_x], [depot_y],
        s=180, marker="*",
        edgecolors="black",
        linewidths=1.0,
        zorder=5
    )

    # vẽ từng route
    for idx, route in enumerate(routes, start=1):
        full_route = [DEPOT_ID] + route + [DEPOT_ID]
        xs = [coords[n][0] for n in full_route]
        ys = [coords[n][1] for n in full_route]

        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2)

        if show_route_id and len(route) > 0:
            mid_node = route[len(route) // 2]
            mx, my = coords[mid_node]
            ax.text(mx, my, str(idx), fontsize=8)

    if show_node_id:
        for node_id, (x, y) in coords.items():
            ax.text(x, y, str(node_id), fontsize=6)

    loads = [route_load(r, demands) for r in routes]
    subtitle = (
        f"Routes = {len(routes)} | "
        f"Distance = {solution_distance(routes, coords):.2f}"
    )

    if capacity is not None:
        over = [i + 1 for i, load in enumerate(loads) if load > capacity]
        if over:
            subtitle += f" | Over capacity: {over}"

    ax.set_title(os.path.basename(route_path) + "\n" + subtitle)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=250)
        print(f"Saved plot to: {save_path}")

    plt.show()


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--instance", required=True, help="File .vrp")
    parser.add_argument("--routes", required=True, help="File routes.txt")
    parser.add_argument("--save", default=None, help="Ảnh output .png")
    parser.add_argument("--show_node_id", action="store_true")
    parser.add_argument("--show_route_id", action="store_true")

    args = parser.parse_args()

    plot_routes(
        instance_path=args.instance,
        route_path=args.routes,
        save_path=args.save,
        show_node_id=args.show_node_id,
        show_route_id=args.show_route_id,
    )


if __name__ == "__main__":
    main()