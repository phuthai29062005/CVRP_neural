import torch

class CVRPenv:
    
    def __init__(self, num_nodes = 20, capacity=None, device=None):
        
        self.num_nodes = num_nodes
        
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        default_capacity = {20: 30.0, 50: 40.0, 100: 50.0}
        self.capacity = float(capacity if capacity is not None else default_capacity.get(num_nodes, 50.0))

        
    def reset(self, batch_size, locs = None, demands = None):
        
        self.batch_size = batch_size
        
        if locs is None or demands is None:
            depot_locs = torch.rand(batch_size, 1, 2, device=self.device)
            customers_locs = torch.rand(batch_size, self.num_nodes, 2, device=self.device)
            self.locs = torch.cat((depot_locs, customers_locs), dim=1)
            
            raw_demands = torch.randint(1, 10, (batch_size, self.num_nodes), device=self.device)
            self.demands = raw_demands.float() / self.capacity   
        else:
            
            self.locs = locs.clone()
            self.demands = demands.clone()
        
        self.current_node = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.remaining_capacity = torch.ones(batch_size, device=self.device) 
        self.remaining_demands = self.demands.clone()     
        
        return self.get_state()
    
    def get_state(self):
        
        depot_demand = torch.zeros(self.batch_size, 1, device=self.device)
        node_demands = torch.cat([depot_demand, self.remaining_demands], dim=1) 
        
        return {
            "locs": self.locs,  
            "demands": node_demands,  
            "current_node": self.current_node,  
            "remaining_capacity": self.remaining_capacity  
        }
        
    def get_mask(self):
        
        """
        mask=True nghĩa là KHÔNG được chọn.
        """
        
        mask = torch.zeros(self.batch_size, self.num_nodes + 1, dtype=torch.bool, device=self.device)
        done_customers = (self.remaining_demands <= 1e-9).all(dim=1)
        
        mask[:, 0] = (self.current_node == 0) & (~done_customers)

        served = self.remaining_demands <= 1e-9
        over_capacity = self.remaining_demands > self.remaining_capacity[:, None]
        mask[:, 1:] = served | over_capacity
        
        mask[done_customers, 0] = False
        mask[done_customers, 1:] = True
        
        return mask
    
    def step(self, next_node):
        """
        next_node: tensor shape [B], giá trị từ 0..n
        """
        next_node = next_node.long()
        B = self.batch_size
        batch_idx = torch.arange(B, device=self.device)

        prev_loc = self.locs[batch_idx, self.current_node]   # [B, 2]
        next_loc = self.locs[batch_idx, next_node]           # [B, 2]
        step_dist = torch.norm(prev_loc - next_loc, dim=1)   # [B]

        is_customer = next_node > 0
        customer_idx = next_node - 1

        # Phục vụ customer
        chosen_demand = torch.zeros(B, device=self.device)
        valid_rows = torch.where(is_customer)[0]
        if len(valid_rows) > 0:
            chosen_demand[valid_rows] = self.remaining_demands[valid_rows, customer_idx[valid_rows]]
            self.remaining_capacity[valid_rows] -= chosen_demand[valid_rows]
            self.remaining_demands[valid_rows, customer_idx[valid_rows]] = 0.0

        # Nếu về depot thì nạp đầy lại xe
        depot_rows = torch.where(~is_customer)[0]
        if len(depot_rows) > 0:
            self.remaining_capacity[depot_rows] = 1.0

        self.current_node = next_node

        all_served = (self.remaining_demands <= 1e-9).all(dim=1)
        done = all_served & (self.current_node == 0)

        reward = -step_dist
        return self.get_state(), reward, done
        
        
        