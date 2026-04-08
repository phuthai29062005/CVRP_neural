from read_data import read_data
import random
import numpy as np
from caculate import get_good_routes, get_fitness


file_path = 'ML4VRP2026/Instances/cvrp/vrp/X-n351-k40.vrp'
dimension, capacity, nodes = read_data(file_path)

print(dimension, capacity)  # Kiểm tra dữ liệu đã đọc đúng chưa

def get_vertices_from_routes(routes):
    """Lấy tất cả đỉnh từ danh sách routes"""
    vertices = set()
    for route in routes:
        for vertex in route:
            vertices.add(vertex)
    return vertices


def select_good_routes_random(good_routes, selection_prob=0.5):
    """Random chọn một số good routes"""
    selected_routes = []
    for route in good_routes:
        if random.random() < selection_prob:
            selected_routes.append(route)
    
    return selected_routes


def remove_duplicate_vertices_in_routes(routes):
    """Loại bỏ routes có đỉnh trùng lặp"""
    kept_routes = []
    used_vertices = set()
    
    for route in routes:
        route_vertices = set(route)
        if route_vertices.isdisjoint(used_vertices):  # Không có đỉnh trùng
            kept_routes.append(route)
            used_vertices.update(route_vertices)
    
    return kept_routes


def GA(parent, route, par1, par2, par3):
    """
    GA Crossover sử dụng good routes
    
    Args:
        parent: Danh sách tất cả parents
        route: Danh sách route markers tương ứng
        par1, par2, par3: Index của 3 parents được chọn
    
    Returns:
        (best_child_permutation, best_child_route, best_child_fitness)
    """
    
    # Lấy 5 good routes từ mỗi parent (15 tổng cộng)
    all_good_routes = []
    for idx in [par1, par2, par3]:
        good_routes, _ = get_good_routes(parent[idx], route[idx], nodes, num_good_routes=5)
        all_good_routes.extend(good_routes)
    
    best_fitness = float('inf')
    best_child = None
    best_child_route = None
    
    # Lặp 5 lần
    for trial in range(5):
        # Random chọn một số good routes (50% xác suất mỗi cái)
        selected_routes = select_good_routes_random(all_good_routes, selection_prob=0.5)
        
        # Loại bỏ routes có đỉnh trùng
        kept_routes = remove_duplicate_vertices_in_routes(selected_routes)
        
        # Lấy tất cả đỉnh đã dùng trong kept_routes
        used_vertices = get_vertices_from_routes(kept_routes)
        
        # Lấy các đỉnh còn lại chưa sử dụng
        all_vertices = set(range(2, dimension + 1))
        remaining_vertices = list(all_vertices - used_vertices)
        
        # Random permutation các đỉnh còn lại
        random.shuffle(remaining_vertices)
        
        # Tạo child permutation: kept_routes + remaining_vertices
        child_permutation = []
        child_route_markers = []
        
        # Thêm các đỉnh từ kept_routes
        current_load = 0
        for i, route_seg in enumerate(kept_routes):
            for j, vertex in enumerate(route_seg):
                child_permutation.append(vertex)
                if j == 0:  # Đỉnh đầu tiên của route
                    child_route_markers.append(1)
                    current_load = nodes[vertex]['demand']
                else:  # Đỉnh tiếp theo
                    child_route_markers.append(0)
                    current_load += nodes[vertex]['demand']
        
        # Thêm các đỉnh còn lại với capacity checking
        for vertex in remaining_vertices:
            # Kiểm tra capacity
            if current_load + nodes[vertex]['demand'] <= capacity:
                child_permutation.append(vertex)
                child_route_markers.append(0)  # Tiếp tục route
                current_load += nodes[vertex]['demand']
            else:
                # Bắt đầu route mới
                child_permutation.append(vertex)
                child_route_markers.append(1)
                current_load = nodes[vertex]['demand']
        
        # Tính fitness cho child này
        child_fitness = get_fitness(child_permutation, child_route_markers, nodes)
        
        # Giữ lại child có fitness tốt nhất
        if child_fitness < best_fitness:
            best_fitness = child_fitness
            best_child = child_permutation
            best_child_route = child_route_markers
    
    return best_child, best_child_route, best_fitness
    
    
    
    
    
    

