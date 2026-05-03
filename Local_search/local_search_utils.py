import math
import numpy as np


ROUTE_PENALTY = 1000.0
DEPOT_ID = 1

def build_distance_matrix(nodes, dimension):
    """
    dist_matrix[a, b] = distance giữa node a và node b.
    Node id chạy từ 1 đến dimension.
    """
    dist_matrix = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)

    for i in range(1, dimension + 1):
        xi, yi = nodes[i]["x"], nodes[i]["y"]

        for j in range(i + 1, dimension + 1):
            xj, yj = nodes[j]["x"], nodes[j]["y"]
            d = math.hypot(xi - xj, yi - yj)

            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def build_k_nearest_neighbors(dist_matrix, dimension, k=10):
    """
    nearest_neighbors[i] = set K node gần nhất của node i.
    """
    nearest_neighbors = {}

    for i in range(1, dimension + 1):
        order = np.argsort(dist_matrix[i])
        neigh = []

        for j in order:
            j = int(j)
            if j == 0 or j == i:
                continue

            neigh.append(j)

            if len(neigh) >= k:
                break

        nearest_neighbors[i] = set(neigh)

    return nearest_neighbors

def euclid(nodes, a, b, dist_matrix=None):
    if dist_matrix is not None:
        return float(dist_matrix[a, b])

    ax, ay = nodes[a]["x"], nodes[a]["y"]
    bx, by = nodes[b]["x"], nodes[b]["y"]
    return math.hypot(ax - bx, ay - by)


def route_distance(route, nodes, dist_matrix=None):
    """
    route: list customer ids, không chứa depot
    """
    if not route:
        return 0.0

    total = euclid(nodes, DEPOT_ID, route[0], dist_matrix)

    for i in range(len(route) - 1):
        total += euclid(nodes, route[i], route[i + 1], dist_matrix)

    total += euclid(nodes, route[-1], DEPOT_ID, dist_matrix)

    return total


def route_demand(route, nodes):
    return sum(nodes[c]["demand"] for c in route)


def rebuild_solution(routes):
    """
    routes: list[list[int]]
    -> parent, route_markers
    """
    parent = []
    route_markers = []

    for r in routes:
        for k, cust in enumerate(r):
            parent.append(cust)
            route_markers.append(1 if k == 0 else 0)

    return parent, route_markers