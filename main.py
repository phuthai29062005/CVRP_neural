import numpy as np
import random
import os

from read_data import read_data
from GA import GA
from caculate import get_route, get_fitness



file_path = 'ML4VRP2026/Instances/cvrp/vrp/X-n351-k40.vrp'
dimension, capacity, nodes = read_data(file_path)
print(dimension, capacity)  # Kiểm tra dữ liệu đã đọc đúng chưa

# Extract instance name from file_path
instance_name = os.path.basename(file_path).split('.')[0]  # X-n351-k40

def get_pop(population):
    
    parent = []
    customers = list(range(2, dimension + 1))
    
    for i in range(population):
        random.shuffle(customers)
        parent.append(customers.copy())
    return parent

def main():
    
    generation = 3000
    population = 100

    parent = get_pop(population)
    route = get_route(parent, dimension, population, capacity, nodes)
    
    fitness = []
    for i in range(population):
        fit = get_fitness(parent[i], route[i], nodes)
        fitness.append((fit, i))
    
    # List to track best fitness each generation
    best_fitness_history = []
    
    for i in range(1, generation):
        
        
        fitness.sort()  # Sắp xếp theo fitness (giá trị nhỏ hơn tốt hơn)
        
        # Track best fitness of this generation
        best_fitness = fitness[0][0]
        print(f"Gen {i}: Best fitness = {best_fitness}")
        best_fitness_history.append(best_fitness)
        # Tạo mapping từ old index -> new index sau sort
        index_mapping = {}
        for new_idx, (fit_val, old_idx) in enumerate(fitness):
            index_mapping[old_idx] = new_idx
        
        new_child = []
        new_fitness = []
        new_route = []
        
        # Elite: Lấy 10% tốt nhất
        for j in range(int(population * 0.1)):
            old_idx = fitness[j][1]
            new_child.append(parent[old_idx])
            new_fitness.append((fitness[j][0], j))
            new_route.append(route[old_idx])
            
        # Crossover: 90% còn lại
        for j in range(int(population * 0.9)):
            
            par1 = np.random.randint(population)
            par2 = np.random.randint(population)
            while par2 == par1:
                par2 = np.random.randint(population)
            
            par3 = np.random.randint(population)
            while par3 == par1 or par3 == par2:
                par3 = np.random.randint(population)
            
            # GA Crossover - sử dụng old indices
            child, child_route, child_fitness = GA(parent, route, par1, par2, par3)
            
            # So sánh child với 3 parents (lấy fitness từ current generation)
            par1_fitness = fitness[index_mapping[par1]][0]
            par2_fitness = fitness[index_mapping[par2]][0]
            par3_fitness = fitness[index_mapping[par3]][0]
            
            # Lấy fitness tốt nhất của 3 parents
            best_parent_fitness = min(par1_fitness, par2_fitness, par3_fitness)
            
            # Nếu child tốt hơn cả 3 parents (fitness nhỏ hơn), lấy child
            if child_fitness < best_parent_fitness:
                new_child.append(child)
                new_route.append(child_route)
                new_fitness.append((child_fitness, len(new_child) - 1))
            else:
                # Nếu không, lấy parent có fitness tốt nhất
                if par1_fitness <= par2_fitness and par1_fitness <= par3_fitness:
                    new_child.append(parent[par1])
                    new_route.append(route[par1])
                    new_fitness.append((par1_fitness, len(new_child) - 1))
                elif par2_fitness <= par3_fitness:
                    new_child.append(parent[par2])
                    new_route.append(route[par2])
                    new_fitness.append((par2_fitness, len(new_child) - 1))
                else:
                    new_child.append(parent[par3])
                    new_route.append(route[par3])
                    new_fitness.append((par3_fitness, len(new_child) - 1))
            
        parent = new_child
        route = new_route
        fitness = new_fitness
    
    # After all generations, save results
    # Sort final fitness to get best solution
    fitness.sort()
    best_idx = fitness[0][1]
    best_parent = parent[best_idx]
    best_route = route[best_idx]
    # Save best fitness history to file
    fitness_file = f"ML4VRP2026/Solutions/cvrp/{instance_name}_fitness.txt"
    os.makedirs(os.path.dirname(fitness_file), exist_ok=True)
    with open(fitness_file, 'w') as f:
        for gen_num, fit_val in enumerate(best_fitness_history, start=1):
            f.write(f"{gen_num}\t{fit_val}\n")
    print(f"Saved fitness history to {fitness_file}")
    
    # Save best routes to file (one route per line) - match BKS format
    route_file = f"ML4VRP2026/Solutions/cvrp/{instance_name}_routes.txt"
    with open(route_file, 'w') as f:
        route_list = []
        current_route = []
        
        # Extract individual routes from best_parent and best_route
        for j, marker in enumerate(best_route):
            if marker == 1 and current_route:
                # Start new route
                route_list.append(current_route.copy())
                current_route = [best_parent[j]]
            else:
                current_route.append(best_parent[j])
        
        # Add last route
        if current_route:
            route_list.append(current_route)
        
        # Write routes in BKS format: Route #1: c1 c2 c3 ...
        for route_num, route_customers in enumerate(route_list, start=1):
            f.write(f"Route #{route_num}: {' '.join(map(str, route_customers))}\n")
    
    print(f"Saved best routes to {route_file}")
    print(f"Best fitness: {fitness[0][0]}")
    print(f"Number of routes: {len(route_list)}")

if __name__ == "__main__":
    main()  