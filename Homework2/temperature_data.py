# Temperature data array from readme.md
# Hours 1 to 24 (military time)

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, fmin
import matplotlib.pyplot as plt

temperatures = [
    75,  # 1
    77,  # 2
    76,  # 3
    73,  # 4
    69,  # 5
    68,  # 6
    63,  # 7
    59,  # 8
    57,  # 9
    55,  # 10
    54,  # 11
    52,  # 12
    50,  # 13
    50,  # 14
    49,  # 15
    49,  # 16
    49,  # 17
    50,  # 18
    54,  # 19
    56,  # 20
    59,  # 21
    63,  # 22
    67,  # 23
    72   # 24
]

if __name__ == "__main__":
    print("Temperature data array:")
    print(temperatures)

    # (a) Parabolic fit
    x = np.arange(1, 25)  # hours 1 to 24
    y = np.array(temperatures)

    # Fit quadratic polynomial: f(x) = A x^2 + B x + C
    coeffs = np.polyfit(x, y, 2)
    A, B, C = coeffs
    print(f"\nParabolic fit coefficients: A={A:.6f}, B={B:.6f}, C={C:.6f}")

    # Evaluate fit at original x
    y_fit = np.polyval(coeffs, x)

    # Calculate E2 error: sum of squared errors
    errors = y - y_fit
    E2 = np.sum(errors**2)
    print(f"E2 error: {E2:.6f}")

    # Evaluate curve for x = 1:0.01:24
    x_eval = np.arange(1, 24.01, 0.01)
    fitted_values = np.polyval(coeffs, x_eval)

    # Save in column vector (numpy array)
    print(f"\nFitted values vector length: {len(fitted_values)}")
    # For brevity, print first and last few values
    print(f"First 10 fitted values: {fitted_values[:10]}")
    print(f"Last 10 fitted values: {fitted_values[-10:]}")

    # (b) Interpolation using INTERP1 and SPLINE
    # Linear interpolation (equivalent to MATLAB interp1 with default linear)
    linear_interp = np.interp(x_eval, x, y)

    # Cubic spline interpolation (equivalent to MATLAB spline)
    cs = CubicSpline(x, y)
    spline_interp = cs(x_eval)

    # Save as column vectors
    print(f"\nLinear interpolation vector length: {len(linear_interp)}")
    print(f"First 10 linear interpolated values: {linear_interp[:10]}")
    print(f"Last 10 linear interpolated values: {linear_interp[-10:]}")

    print(f"\nSpline interpolation vector length: {len(spline_interp)}")
    print(f"First 10 spline interpolated values: {spline_interp[:10]}")
    print(f"Last 10 spline interpolated values: {spline_interp[-10:]}")

    # Plot all results
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'bo', markersize=6, label='Original temperature data')
    plt.plot(x_eval, fitted_values, 'r-', linewidth=2, label='Parabolic fit')
    plt.plot(x_eval, linear_interp, 'g--', linewidth=2, label='Linear interpolation')
    plt.plot(x_eval, spline_interp, 'm-.', linewidth=2, label='Cubic spline interpolation')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°F)')
    plt.title('Temperature Data: Fitting and Interpolation Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temperature_analysis.png', dpi=300, bbox_inches='tight')
    print("\nTemperature analysis plot saved as 'temperature_analysis.png'")
    # plt.show()  # Uncomment to display plot

    # (c) Least-squares fit: y = A*cos(B*x) + C using fmin
    def temp_func(x, A, B, C):
        return A * np.cos(B * x) + C

    def objective(params):
        A, B, C = params
        y_pred = temp_func(x, A, B, C)
        return np.sum((y - y_pred)**2)

    # Initial guesses - critical for convergence!
    p0 = [10, 0.5, 60]

    # A variants
    # p1 = [5, 0.5, 60]
    # p2 = [20, 0.5, 60]

    # B variants
    # p1 = [10, 0.1, 60]
    # p2 = [10, 1.0, 60]

    # C variants
    p1 = [10, 0.1, 0]
    p2 = [10, 0.1, 100]

    initial_guesses = [p0, p1, p2]
    labels = ['p0', 'p1', 'p2']
    results = []
    
    # Try each initial guess
    print(f"\nLeast-squares fit using fmin: y = A*cos(B*x) + C")
    for init_guess, label in zip(initial_guesses, labels):
        popt = fmin(objective, init_guess, xtol=1e-6, ftol=1e-6, maxiter=10000, disp=False)
        A_fit, B_fit, C_fit = popt
        
        # Calculate E2 error
        y_fit_cos = temp_func(x, A_fit, B_fit, C_fit)
        errors_cos = y - y_fit_cos
        E2_cos = np.sum(errors_cos**2)
        
        results.append((A_fit, B_fit, C_fit, E2_cos))
        
        print(f"\n{label}: A = {A_fit:.6f}, B = {B_fit:.6f}, C = {C_fit:.6f}, E2 = {E2_cos:.6f}")
    
    # Choose the best result (lowest E2 error)
    best_idx = np.argmin([r[3] for r in results])
    A_fit, B_fit, C_fit, E2_cos = results[best_idx]
    print(f"\nBest fit: {labels[best_idx]} with E2 error = {E2_cos:.6f}")

    # Evaluate curve for x = 1:0.01:24
    x_eval_cos = np.arange(1, 24.01, 0.01)
    fitted_values_cos = temp_func(x_eval_cos, A_fit, B_fit, C_fit)

    print(f"\nFitted values vector (A6) length: {len(fitted_values_cos)}")
    print(f"First 10 fitted values: {fitted_values_cos[:10]}")
    print(f"Last 10 fitted values: {fitted_values_cos[-10:]}")

    # Plot all three fits
    plt.figure(figsize=(12, 7))
    plt.plot(x, y, 'bo', markersize=6, label='Original temperature data')
    
    colors = ['red', 'green', 'orange']
    for i, (A, B, C, E2) in enumerate(results):
        fit_values = temp_func(x_eval_cos, A, B, C)
        plt.plot(x_eval_cos, fit_values, colors[i], linestyle='--' if i > 0 else '-', 
                linewidth=2, label=f'{labels[i]}: $y = {A:.2f}\\cos({B:.2f}x) + {C:.2f}$ (E2={E2:.1f})')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°F)')
    plt.title('Temperature Data: Cosine Least-Squares Fit using fmin with Different Initial Guesses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temperature_cosine_fit_fmin_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'temperature_cosine_fit_fmin_comparison.png'")