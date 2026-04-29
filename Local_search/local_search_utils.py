import math


ROUTE_PENALTY = 1000.0
DEPOT_ID = 1


def euclid(nodes, a, b):
    ax, ay = nodes[a]["x"], nodes[a]["y"]
    bx, by = nodes[b]["x"], nodes[b]["y"]
    return math.hypot(ax - bx, ay - by)


def route_distance(route, nodes):
    """
    route: list customer ids, không chứa depot
    """
    if not route:
        return 0.0

    total = euclid(nodes, DEPOT_ID, route[0])
    for i in range(len(route) - 1):
        total += euclid(nodes, route[i], route[i + 1])
    total += euclid(nodes, route[-1], DEPOT_ID)
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