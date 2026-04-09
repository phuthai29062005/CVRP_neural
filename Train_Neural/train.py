import torch
import torch.optim as optim
import numpy as np
import time
import random

from cvrp_model import CVRPModel
from cvrp_env import CVRPEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# HYPERPARAM
# =====================
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-4

# =====================
# MODEL
# =====================
model = CVRPModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# =====================
# TRAIN LOOP
# =====================
for epoch in range(EPOCHS):
    start_time = time.time()
    
    epoch_loss = 0
    
    for batch_id in range(1000):   # giảm xuống cho nhanh
        
        # 🔥 MULTI-SIZE TRAIN (QUAN TRỌNG)
        num_nodes = random.choice([100, 120, 150])
        
        env = CVRPEnv(batch_size=BATCH_SIZE, num_nodes=num_nodes)
        state = env.reset()
        
        log_probs = []
        rewards = None
        
        done = torch.zeros(BATCH_SIZE, dtype=torch.bool).to(device)
        
        while not done.all():
            
            # forward
            probs = model(state)
            
            # sample
            selected = probs.multinomial(1).squeeze(-1)
            
            log_prob = torch.log(
                probs.gather(1, selected.unsqueeze(1)).squeeze(1) + 1e-8
            )
            
            log_probs.append(log_prob)
            
            # step
            state, reward, done = env.step(selected)
        
        # total reward
        rewards = reward   # distance
        
        # stack log probs
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
        
        # REINFORCE loss
        loss = (rewards * log_probs).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Time: {time.time()-start_time:.2f}s")

    # save mỗi epoch
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

# save final
torch.save(model.state_dict(), "model_final.pt")