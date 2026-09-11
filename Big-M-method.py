import numpy as np
M = 10000.0  
cj = np.array([2.0, 3.0, 0.0, 0.0, M, M])
tableau = np.array([
    [20.0, 10.0, -1.0,  0.0, 1.0, 0.0, 1000.0],
    [10.0, 30.0,  0.0, -1.0, 0.0, 1.0, 1200.0]
])

basic_vars = [4, 5]

print("--- Starting Big-M Simplex Optimization ---")

iteration = 0
while True:
    iteration += 1
    print(f"\nIteration {iteration}:")
    cb = cj[basic_vars]
    
    zj = np.dot(cb, tableau[:, :-1])
    
    zj_cj = zj - cj
    
    if np.all(zj_cj <= 1e-7):
        print("Optimality reached! All Z_j - C_j <= 0.")
        break
        
    entering_idx = np.argmax(zj_cj)
    print(f"Entering Variable Index: {entering_idx}")
    
    ratios = []
    for i in range(tableau.shape[0]):
        if tableau[i, entering_idx] > 0:
            ratios.append(tableau[i, -1] / tableau[i, entering_idx])
        else:
            ratios.append(np.inf) 
            
    if np.all(np.isinf(ratios)):
        raise ValueError("The solution is unbounded.")
        
    leaving_row = np.argmin(ratios)
    leaving_var = basic_vars[leaving_row]
    print(f"Leaving Variable Index: {leaving_var} (Row {leaving_row})")
    
    basic_vars[leaving_row] = entering_idx
    
    pivot_element = tableau[leaving_row, entering_idx]
    
    tableau[leaving_row, :] = tableau[leaving_row, :] / pivot_element
    
    for i in range(tableau.shape[0]):
        if i != leaving_row:
            multiplier = tableau[i, entering_idx]
            tableau[i, :] = tableau[i, :] - (multiplier * tableau[leaving_row, :])

optimal_x = np.zeros(len(cj))

for i in range(len(basic_vars)):
    var_idx = basic_vars[i]
    optimal_x[var_idx] = tableau[i, -1]

min_cost = np.dot(optimal_x, cj)

print("\n--- Final Optimization Results ---")
print(f"Optimal x1 (Supplement 1): {optimal_x[0]:.2f}")
print(f"Optimal x2 (Supplement 2): {optimal_x[1]:.2f}")
print(f"Minimum Total Cost (Z): {min_cost:.2f} INR")