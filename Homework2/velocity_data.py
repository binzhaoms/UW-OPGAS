# Velocity data as a function of time (seconds)
# Time from 0 to 30 seconds in steps of one second

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

v = [30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50,
     49, 46, 48, 50, 53, 55, 54, 53]

def velocity_func(t, A, B, C, D):
    return A * np.cos(B * t) + C * t + D

if __name__ == "__main__":
    print("Velocity data array:")
    print(v)

    # Time array
    t = np.arange(0, 31)  # 0 to 30 inclusive
    v_array = np.array(v)

    # Initial guesses
    p0 = [3, np.pi/4, 2/3, 32]
    p1 = [0, 0, 0, 0]
    p2 = [6, np.pi/4, 2/3, 32]
    p3 = [3, np.pi/2, 2/3, 32]
    p4 = [3, np.pi/4, 4/3, 32]
    p5 = [3, np.pi/4, 2/3, 64]
    p6= [0, np.pi/4, 0, 0]
    p7 = [100, np.pi/4, 100, 100]
    
    # List of all initial guesses
    initial_guesses = [p0, p1, p2, p3, p4, p5, p6, p7]
    labels = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
    
    # Store fitted parameters for each
    fitted_params = []
    
    for i, p_init in enumerate(initial_guesses):
        try:
            popt, pcov = curve_fit(velocity_func, t, v_array, p0=p_init)
            A_fit, B_fit, C_fit, D_fit = popt
            fitted_params.append((A_fit, B_fit, C_fit, D_fit))
            
            # Calculate fitted values at original points
            y_fit = velocity_func(t, A_fit, B_fit, C_fit, D_fit)
            # Calculate E2 error
            errors = v_array - y_fit
            E2 = np.sum(errors**2)
            
            print(f"\nFitted parameters (with {labels[i]}):")
            print(f"A = {A_fit:.6f}")
            print(f"B = {B_fit:.6f}")
            print(f"C = {C_fit:.6f}")
            print(f"D = {D_fit:.6f}")
            print(f"E2 error = {E2:.6f}")
        except Exception as e:
            print(f"\nFitting with {labels[i]} failed: {e}")
            fitted_params.append(None)

    # Generate evaluation time array
    t_eval = np.arange(0, 30.01, 0.01)

    # Visualize all the fits
    plt.figure(figsize=(14, 10))
    plt.plot(t, v_array, 'bo', markersize=4, label='Original data points')
    
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, params in enumerate(fitted_params):
        if params is not None:
            A, B, C, D = params
            fitted_values = velocity_func(t_eval, A, B, C, D)
            plt.plot(t_eval, fitted_values, 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)], 
                    linewidth=2,
                    label=f'Fit with {labels[i]}: $A={A:.2f}, B={B:.2f}, C={C:.2f}, D={D:.2f}$')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time: Comparison of Least Squares Fits with Different Initial Guesses')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('velocity_fit_all_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot with all fits saved as 'velocity_fit_all_comparison.png'")
    # plt.show()  # Commented out to avoid blocking