import numpy as np
import random

def get_route(parent, dimension, population, capacity, nodes):
    """
    Tạo route markers cho mỗi parent
    
    Args:
        parent: Danh sách parents (mỗi parent là hoán vị khách hàng từ 2-101)
        dimension: 101 (bao gồm depot)
        population: Số lượng parents
        capacity: Dung tích xe
        nodes: Dictionary chứa info các nút
    
    Returns:
        Danh sách route markers [1, 0, 0, 1, 0, ...]
    """
    route = []
    for i in range(population):
        permutation = parent[i].copy()  # [2, 3, 4, ..., 101]
        first_route = []
        current_load = nodes[permutation[0]]['demand']
        first_route.append(1)  # Bắt đầu route mới với khách hàng đầu tiên
        
        for j in range(1, len(permutation)):
            if current_load + nodes[permutation[j]]['demand'] <= capacity:
                first_route.append(0)  # Tiếp tục route hiện tại
                current_load += nodes[permutation[j]]['demand']
            else:
                first_route.append(1)  # Bắt đầu route mới
                current_load = nodes[permutation[j]]['demand']
        
        route.append(first_route)
    return route

def get_fitness(parent, route, nodes):
    """
    Tính fitness (1000*num_routes + total_distance)
    
    Args:
        parent: Hoán vị khách hàng [2, 3, 4, ..., 101]
        route: Route markers [1, 0, 0, 1, 0, ...]
        nodes: Dictionary chứa info các nút
    
    Returns:
        fitness = 1000 * num_routes + total_distance
    """
    # Tách các route riêng lẻ
    routes = separate_routes(parent, route)
    
    # Tính total distance từ tất cả routes
    total_distance = 0
    for route_customers in routes:
        total_distance += calculate_route_distance(route_customers, nodes)
    
    num_routes = len(routes)
    fitness = 1000 * num_routes + total_distance
    return fitness


def separate_routes(parent, route):
    """
    Tách các route riêng lẻ từ parent và route marker
    
    Args:
        parent: Danh sách khách hàng [c1, c2, c3, ...]
        route: Danh sách marker [1, 0, 0, 1, 0, 1, ...] (1 = bắt đầu route mới)
    
    Returns:
        Danh sách các route: [[c1, c2], [c3], [c4, c5, ...]]
    """
    routes = []
    current_route = []
    
    for i in range(len(parent)):
        if route[i] == 1:  # Bắt đầu route mới
            if current_route:  # Nếu có route trước đó, lưu lại
                routes.append(current_route)
            current_route = [parent[i]]
        else:  # Tiếp tục route hiện tại
            current_route.append(parent[i])
    
    # Lưu route cuối cùng
    if current_route:
        routes.append(current_route)
    
    return routes


def calculate_route_distance(route_customers, nodes):
    """
    Tính tổng khoảng cách của 1 route
    
    Args:
        route_customers: Danh sách khách hàng trong route [c1, c2, c3, ...]
        nodes: Dictionary chứa tọa độ các nút
    
    Returns:
        Tổng khoảng cách (không tính đi về depot)
    """
    distance = 0
    
    # Tính khoảng cách từ depot (node 1) đến khách hàng đầu tiên
    distance += np.sqrt((nodes[route_customers[0]]['x'] - nodes[1]['x'])**2 + 
                       (nodes[route_customers[0]]['y'] - nodes[1]['y'])**2)
    
    # Tính khoảng cách giữa các khách hàng trong route
    for i in range(1, len(route_customers)):
        distance += np.sqrt((nodes[route_customers[i]]['x'] - nodes[route_customers[i-1]]['x'])**2 + 
                           (nodes[route_customers[i]]['y'] - nodes[route_customers[i-1]]['y'])**2)
    
    # Tính khoảng cách từ khách hàng cuối cùng về depot (node 1)
    distance += np.sqrt((nodes[route_customers[-1]]['x'] - nodes[1]['x'])**2 + 
                       (nodes[route_customers[-1]]['y'] - nodes[1]['y'])**2)
    
    return distance


def get_good_routes(parent, route, nodes, num_good_routes=5):
    """
    Lấy các route tốt nhất (có score thấp nhất) từ 1 parent
    
    Args:
        parent: Hoán vị khách hàng [c1, c2, c3, ...]
        route: Danh sách marker [1, 0, 0, 1, 0, 1, ...]
        nodes: Dictionary chứa tọa độ các nút
        num_good_routes: Số route tốt nhất cần lấy (mặc định 5)
    
    Returns:
        Danh sách 5 route tốt nhất: [[route1], [route2], ...]
        Danh sách score tương ứng: [score1, score2, ...]
    """
    # Tách các route riêng lẻ
    routes = separate_routes(parent, route)
    
    # Tính score cho mỗi route
    route_scores = []
    for r in routes:
        distance = calculate_route_distance(r, nodes)
        num_customers = len(r)
        # Score = (1000 + distance) / số khách hàng
        score = (1000 + distance) / num_customers
        route_scores.append((score, r))
    
    # Sắp xếp theo score (tăng dần) và lấy 5 route có score thấp nhất
    route_scores.sort(key=lambda x: x[0])
    good_routes_with_scores = route_scores[:num_good_routes]
    
    # Tách riêng danh sách route và score
    good_routes = [r[1] for r in good_routes_with_scores]
    good_scores = [r[0] for r in good_routes_with_scores]
    
    return good_routes, good_scores


def get_all_good_routes(parents, routes, nodes, num_good_routes=5):
    """
    Lấy các route tốt nhất từ tất cả parents
    
    Args:
        parents: Danh sách tất cả parent (hoán vị)
        routes: Danh sách route marker tương ứng
        nodes: Dictionary chứa tọa độ các nút
        num_good_routes: Số route tốt nhất từ mỗi parent
    
    Returns:
        Danh sách tất cả good routes: [[route1], [route2], ...]
    """
    all_good_routes = []
    
    for i in range(len(parents)):
        good_routes, good_scores = get_good_routes(parents[i], routes[i], nodes, num_good_routes)
        all_good_routes.extend(good_routes)
    
    return all_good_routes