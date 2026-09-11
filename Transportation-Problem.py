import numpy as np

def get_vam_allocation(cost, supply, demand):
    rows, cols = cost.shape
    allocation = np.zeros((rows, cols))
    s_copy, d_copy, c_copy = supply.copy(), demand.copy(), cost.copy()
    
    while np.sum(s_copy) > 0 and np.sum(d_copy) > 0:
        row_pen, col_pen = [], []
        for i in range(rows):
            if s_copy[i] > 0:
                v = np.sort([c_copy[i][j] for j in range(cols) if d_copy[j] > 0])
                row_pen.append(v[1] - v[0] if len(v) > 1 else v[0])
            else: row_pen.append(-1)
                
        for j in range(cols):
            if d_copy[j] > 0:
                v = np.sort([c_copy[i][j] for i in range(rows) if s_copy[i] > 0])
                col_pen.append(v[1] - v[0] if len(v) > 1 else v[0])
            else: col_pen.append(-1)
                
        if max(row_pen) >= max(col_pen):
            r = row_pen.index(max(row_pen))
            c = min([j for j in range(cols) if d_copy[j] > 0], key=lambda j: c_copy[r][j])
        else:
            c = col_pen.index(max(col_pen))
            r = min([i for i in range(rows) if s_copy[i] > 0], key=lambda i: c_copy[i][c])
            
        qty = min(s_copy[r], d_copy[c])
        allocation[r][c] = qty
        s_copy[r] -= qty
        d_copy[c] -= qty
    return allocation

def get_modi_modifiers(allocation, cost):
    rows, cols = cost.shape
    u, v = np.full(rows, np.nan), np.full(cols, np.nan)
    u[0] = 0
    while np.isnan(u).any() or np.isnan(v).any():
        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] > 0:
                    if not np.isnan(u[i]) and np.isnan(v[j]): v[j] = cost[i][j] - u[i]
                    elif np.isnan(u[i]) and not np.isnan(v[j]): u[i] = cost[i][j] - v[j]
    return u, v

def get_closed_loop(start_pos, allocation):
    rows, cols = allocation.shape
    path = [start_pos]
    
    def dfs(curr_pos, is_vertical, visited):
        if len(visited) > 3 and curr_pos == start_pos: return True
        r, c = curr_pos
        
        if is_vertical:
            for next_r in range(rows):
                if next_r != r and (allocation[next_r][c] > 0 or (next_r, c) == start_pos):
                    if (next_r, c) not in visited or (next_r, c) == start_pos:
                        path.append((next_r, c))
                        if dfs((next_r, c), not is_vertical, visited + [(next_r, c)]): return True
                        path.pop()
        else:
            for next_c in range(cols):
                if next_c != c and (allocation[r][next_c] > 0 or (r, next_c) == start_pos):
                    if (r, next_c) not in visited or (r, next_c) == start_pos:
                        path.append((r, next_c))
                        if dfs((r, next_c), not is_vertical, visited + [(r, next_c)]): return True
                        path.pop()
        return False
    
    dfs(start_pos, False, [start_pos]) 
    if len(path) == 1: dfs(start_pos, True, [start_pos]) 
    return path[:-1] 

cost = np.array([[10, 2, 20, 11], [12, 7, 9, 20], [4, 14, 16, 18]])
supply = np.array([15, 25, 10])
demand = np.array([5, 15, 15, 15])

alloc = get_vam_allocation(cost, supply, demand)
print("Initial VAM Allocation:\n", alloc)
print("VAM Cost:", np.sum(alloc * cost), "\n")

iteration = 1
while True:
    print(f"--- MODI Iteration {iteration} ---")
    u, v = get_modi_modifiers(alloc, cost)
    
    delta = np.zeros(cost.shape)
    min_delta, entering_cell = 0, None
    
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            if alloc[i][j] == 0:
                delta[i][j] = cost[i][j] - (u[i] + v[j])
                if delta[i][j] < min_delta:
                    min_delta = delta[i][j]
                    entering_cell = (i, j)
                    
    if min_delta >= 0:
        print("All Delta >= 0. Solution is OPTIMAL.")
        break
        
    print(f"Negative Delta found at {entering_cell}. Shifting allocations...")
    loop = get_closed_loop(entering_cell, alloc)
    
    minus_cells = [loop[i] for i in range(1, len(loop), 2)]
    theta = min([alloc[r][c] for r, c in minus_cells])
    
    for i, (r, c) in enumerate(loop):
        if i % 2 == 0: alloc[r][c] += theta
        else: alloc[r][c] -= theta
        
    print("New Allocation Matrix:\n", alloc)
    print("Current Cost:", np.sum(alloc * cost), "\n")
    iteration += 1

print("\nFINAL OPTIMAL COST:", np.sum(alloc * cost))