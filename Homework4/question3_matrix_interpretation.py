"""
Question 3: Interpretation of U, Σ, and V Matrices

Task: Explain the interpretation of the U, Σ, and V matrices 
from the Singular Value Decomposition (SVD) of MNIST data.

For matrix A = U * Σ * V^T:
- U: Left singular vectors (spatial basis/eigenfaces)
- Σ: Singular values (importance/energy of each mode)
- V^T: Right singular vectors (image coefficients)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

print("=" * 70)
print("QUESTION 3: INTERPRETATION OF U, Σ, AND V MATRICES")
print("=" * 70)

# Load SVD components
print("\n[1] Loading SVD components...")
U = np.load('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/U_svd.npy')
sigma = np.load('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/sigma_svd.npy')
Vt = np.load('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/Vt_svd.npy')
labels = np.load('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/mnist_labels.npy')

print(f"    U shape: {U.shape} (784 pixels × 784 modes)")
print(f"    Σ shape: {sigma.shape} (784 singular values)")
print(f"    V^T shape: {Vt.shape} (784 modes × 70000 images)")

print("\n" + "=" * 70)
print("DETAILED MATRIX INTERPRETATIONS")
print("=" * 70)

# ============ U MATRIX INTERPRETATION ============
print("\n[2] U MATRIX INTERPRETATION:")
print("    " + "-" * 66)
print(f"""
    Shape: 784 × 784
    
    WHAT U REPRESENTS:
    - Contains the LEFT SINGULAR VECTORS (orthonormal basis)
    - Each column u_i is a basis vector in PIXEL SPACE
    - Each u_i is a 784-dimensional vector (can be reshaped to 28×28)
    
    INTERPRETATION AS EIGENFACES/EIGENMODES:
    - Column 1 (u_1): First principal component pattern
    - Column 2 (u_2): Second principal component pattern
    - Column i (u_i): i-th most important pixel pattern
    
    PHYSICAL MEANING:
    - u_i represents spatial patterns/features common across all digits
    - u_1 captures the MOST important spatial variation in digits
    - Later columns (u_k) capture noise and fine details
    - These are like "prototype shapes" or "basis patterns"
    
    MATHEMATICAL PROPERTY:
    - U^T * U = I (orthonormal columns)
    - Columns are orthogonal to each other
    - Each column has unit norm (length = 1)
    
    HOW TO USE U:
    - U acts as a transformation matrix from image space to mode space
    - If we project an image x onto U: coefficients = U^T * x
    - Shows "how much" of each basis pattern is in the image
""")

# ============ SIGMA MATRIX INTERPRETATION ============
print("\n[3] Σ (SIGMA) MATRIX INTERPRETATION:")
print("    " + "-" * 66)
print(f"""
    Shape: 784 singular values (σ_1, σ_2, ..., σ_784)
    Mathematical form: Diagonal matrix with singular values on diagonal
    
    WHAT Σ REPRESENTS:
    - Importance/energy/magnitude of each basis mode
    - Larger σ_i = that mode explains more variance in data
    
    SPECIFIC VALUES (from MNIST):
    - σ_1 = {sigma[0]:.2f} (LARGEST - most important)
    - σ_2 = {sigma[1]:.2f}
    - σ_10 = {sigma[9]:.2f}
    - σ_100 = {sigma[99]:.2f}
    - σ_784 = {sigma[-1]:.6f} (SMALLEST - almost no variance)
    
    INTERPRETATION:
    - σ_i represents the "strength" of the i-th pattern in U
    - The ratio σ_i / σ_1 shows relative importance
    - Small σ values can often be discarded (dimensionality reduction)
    
    CONNECTION TO VARIANCE:
    - Variance explained by mode i = σ_i² / (sum of all σ²)
    - Total energy = Σ σ_i²
    - σ_1² alone accounts for {(sigma[0]**2 / np.sum(sigma**2))*100:.2f}% of total variance!
    - Top 102 modes account for 95% of variance
    
    SCALING INTERPRETATION:
    - If we think of U as basis vectors (unit length)
    - Σ scales these basis vectors by their importance
    - Large σ = that basis direction has high variance
    - Small σ = that basis direction has low variance
    
    SIGNAL VS NOISE:
    - Large singular values: signal/structure
    - Small singular values: noise/fine details
    - We can filter by threshold (keep only large σ)
""")

# ============ V^T MATRIX INTERPRETATION ============
print("\n[4] V^T (V-TRANSPOSE) MATRIX INTERPRETATION:")
print("    " + "-" * 66)
print(f"""
    Shape: 784 × 70000
    - Rows: 784 (one for each mode/basis pattern)
    - Columns: 70000 (one for each image in dataset)
    
    WHAT V^T REPRESENTS:
    - Contains the RIGHT SINGULAR VECTORS (coefficients)
    - Each column is an image in terms of its mode coefficients
    - Entry V_ij^T = coefficient of mode i in image j
    
    INTERPRETATION:
    - V^T^T = V is the matrix we typically work with (784 × 70000)
    - V columns are orthonormal (V^T * V = I)
    - V_i = i-th row of V^T = how all 70000 images project onto mode i
    
    SPECIFIC MEANING:
    - V_{{1,j}} = coefficient of mode 1 in image j SCALED by σ_1
    - V_{{2,j}} = coefficient of mode 2 in image j SCALED by σ_2
    - V_{{i,j}} = how much image j uses basis pattern u_i
    
    PROJECTION/RECONSTRUCTION:
    - To reconstruct image j: x_j = Σ_i (σ_i * u_i * V_{{i,j}})
    - V coefficients are like "recipe" showing which bases to mix
    - Large V_{{i,j}} = image j strongly uses pattern u_i
    - Small V_{{i,j}} = image j barely uses pattern u_i
    
    FOR CLUSTERING/CLASSIFICATION:
    - V coefficients ARE the features for machine learning!
    - These are PCA (Principal Component Analysis) coordinates
    - Top 102 coefficients give 95% variance reduction
    - Can use just rows 1-102 of V^T instead of all 784
    
    DIGIT SEPARATION IN V SPACE:
    - Different digits occupy different regions of V coefficient space
    - Rows 1-10 of V^T may separate well by digit
    - This is why LDA/SVM classifiers work on (U * Σ) features
""")

# ============ COMBINED INTERPRETATION ============
print("\n[5] COMBINED INTERPRETATION (A = U * Σ * V^T):")
print("    " + "-" * 66)
print(f"""
    DATA MATRIX RECONSTRUCTION:
    - Original data matrix A: 784 × 70000 (reshape of MNIST images)
    - A = U @ diag(Σ) @ V^T
    - A_{{ij}} = pixel intensity at position i in image j
    
    DECOMPOSITION PROCESS:
    1. U matrix: Lists all spatial patterns/basis shapes (28×28 "faces")
    2. Σ values: Scales each pattern by its importance
    3. V^T rows: Shows which patterns are present in each image
    
    MENTAL MODEL - "Recipe for Images":
    - U = Available ingredients (28×28 pixel patterns)
    - Σ = How much of each ingredient is typically needed
    - V^T = Recipe for each specific image (which ingredients + amounts)
    
    DIMENSIONALITY REDUCTION:
    - Use only top r modes (largest σ values)
    - Keep: U[:, :r], Σ[:r], V^T[:r, :]
    - This captures 95% info with 13% dimensions
    - Reconstruction A_approx = U[:,1:r] @ diag(Σ[1:r]) @ V^T[1:r,:]
    
    RANK INTERPRETATION:
    - Rank of A = number of non-zero singular values = {np.sum(sigma > 1e-10)}
    - Mathematical rank: min(784, 70000) but actual rank is {np.sum(sigma > 1e-10)}
    - Effective rank (95% variance): ~102
    - Data is COMPRESSIBLE because rank << 70000
""")

# ============ VISUALIZATION ============
print("\n[6] Creating visualizations...")

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# ---- U MATRIX VISUALIZATION ----
print("    Creating U matrix eigenface visualizations...")

# Plot 1: First 9 U basis vectors (eigenfaces)
for idx in range(9):
    ax = fig.add_subplot(gs[0, idx // 3 + 0])
    if idx == 0:
        eigenface = U[:, idx].reshape(28, 28)
        im = ax.imshow(eigenface, cmap='RdBu_r', vmin=-0.15, vmax=0.15)
        ax.set_title(f'U Mode 1\n(σ₁={sigma[0]:.0f})', fontweight='bold', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    elif idx == 1:
        eigenface = U[:, idx].reshape(28, 28)
        im = ax.imshow(eigenface, cmap='RdBu_r', vmin=-0.15, vmax=0.15)
        ax.set_title(f'U Mode 2\n(σ₂={sigma[1]:.0f})', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    elif idx == 2:
        eigenface = U[:, idx].reshape(28, 28)
        im = ax.imshow(eigenface, cmap='RdBu_r', vmin=-0.15, vmax=0.15)
        ax.set_title(f'U Mode 3\n(σ₃={sigma[2]:.0f})', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

# Plot eigenfaces grid differently
ax_eig = fig.add_subplot(gs[0, 1])
modes_grid = np.zeros((84, 84))
for i in range(3):
    for j in range(3):
        mode_idx = i * 3 + j
        eigenface = U[:, mode_idx].reshape(28, 28)
        # Normalize for display
        eigenface_norm = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min() + 1e-8)
        modes_grid[i*28:i*28+28, j*28:j*28+28] = eigenface_norm

ax_eig.imshow(modes_grid, cmap='gray')
ax_eig.set_title('First 9 U Eigenmodes\n(Left Singular Vectors)', fontweight='bold', fontsize=11)
ax_eig.set_xticks([])
ax_eig.set_yticks([])

# Add labels
ax_eig.text(14, -2, '1', ha='center', fontsize=9, fontweight='bold')
ax_eig.text(42, -2, '2', ha='center', fontsize=9, fontweight='bold')
ax_eig.text(70, -2, '3', ha='center', fontsize=9, fontweight='bold')

# ---- SIGMA MATRIX VISUALIZATION ----
print("    Creating Σ matrix visualizations...")

# Plot 2: Sigma values (log scale)
ax = fig.add_subplot(gs[0, 2])
ax.semilogy(range(1, 101), sigma[:100], 'ro-', linewidth=2, markersize=4)
ax.fill_between(range(1, 101), sigma[:100], alpha=0.3, color='red')
ax.set_xlabel('Mode Index (i)', fontsize=10)
ax.set_ylabel('Singular Value σᵢ (log scale)', fontsize=10)
ax.set_title('Σ Matrix: Singular Values\n(Importance/Energy)', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, which='both')

# Plot 3: Sigma relative importance
ax = fig.add_subplot(gs[1, 0])
relative_sigma = sigma / sigma[0] * 100
ax.plot(range(1, 101), relative_sigma[:100], 'b-', linewidth=2)
ax.fill_between(range(1, 101), relative_sigma[:100], alpha=0.3, color='blue')
ax.set_xlabel('Mode Index (i)', fontsize=10)
ax.set_ylabel('Relative Importance (%)', fontsize=10)
ax.set_title('Σ Matrix: Relative Importance\n(% of σ₁)', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 4: Energy distribution
ax = fig.add_subplot(gs[1, 1])
energy_percent = (sigma**2 / np.sum(sigma**2)) * 100
cumulative_energy = np.cumsum(energy_percent)
ax.bar(range(1, 26), energy_percent[:25], color='steelblue', edgecolor='black', label='Mode energy')
ax.plot(range(1, 26), cumulative_energy[:25], 'r-o', linewidth=2, markersize=5, label='Cumulative')
ax.set_xlabel('Mode Index (i)', fontsize=10)
ax.set_ylabel('Energy / Cumulative (%)', fontsize=10)
ax.set_title('Σ Matrix: Energy Distribution\n(First 25 modes)', fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# ---- V^T MATRIX VISUALIZATION ----
print("    Creating V^T matrix visualizations...")

# Plot 5: V^T coefficients for sample images
ax = fig.add_subplot(gs[1, 2])
# Show first 50 modes for 10 sample images (one per digit 0-9)
sample_indices = [np.where(labels == d)[0][0] for d in range(10)]
V_sample = Vt[:50, sample_indices]
im = ax.imshow(V_sample, cmap='RdBu_r', aspect='auto', vmin=-20, vmax=20)
ax.set_xlabel('Sample Image (Digit 0-9)', fontsize=10)
ax.set_ylabel('Mode Index (i)', fontsize=10)
ax.set_title('V^T Matrix Values\n(Image Coefficients)', fontweight='bold', fontsize=11)
ax.set_xticks(range(10))
ax.set_xticklabels(range(10))
plt.colorbar(im, ax=ax, label='Coefficient Value')

# Plot 6: Image reconstruction breakdown
ax = fig.add_subplot(gs[2, 0])
test_digit = 3
test_idx = np.where(labels == test_digit)[0][0]
# Show contribution of each mode to reconstruction
mode_contributions = np.abs(sigma[:100] * Vt[:100, test_idx])
ax.bar(range(1, 101), mode_contributions, color='darkgreen', edgecolor='black', linewidth=0.3)
ax.set_xlabel('Mode Index (i)', fontsize=10)
ax.set_ylabel('Mode Contribution\n|σᵢ × Vᵢⱼ|', fontsize=10)
ax.set_title(f'Mode Contributions to Image Reconstruction\n(Test digit: {test_digit})', 
             fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Plot 7: U * Sigma interpretation
ax = fig.add_subplot(gs[2, 1])
# Show first 5 scaled eigenmodes (U * Sigma)
US_modes = np.zeros((84, 28))
for i in range(3):
    scaled_mode = U[:, i] * sigma[i]
    scaled_mode = scaled_mode.reshape(28, 28)
    scaled_norm = (scaled_mode - scaled_mode.min()) / (scaled_mode.max() - scaled_mode.min() + 1e-8)
    US_modes[i*28:i*28+28, :] = scaled_norm

ax.imshow(US_modes, cmap='gray')
ax.set_title('U × Σ Interpretation\n(Scaled Eigenmodes)', fontweight='bold', fontsize=11)
ax.set_xticks([])
ax.set_yticks([])
ax.text(-5, 14, 'σ₁u₁', ha='right', fontsize=9, fontweight='bold')
ax.text(-5, 42, 'σ₂u₂', ha='right', fontsize=9, fontweight='bold')
ax.text(-5, 70, 'σ₃u₃', ha='right', fontsize=9, fontweight='bold')

# Plot 8: Reconstruction process visualization
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

summary_text = f"""
MATRIX ROLES IN A = U·Σ·V^T

┌─────────────────────────────────────────┐
│ U (784 × 784): SPATIAL BASIS            │
│ • Eigenfaces/pixel patterns              │
│ • Orthonormal columns (unit length)      │
│ • Each column = one basis pattern        │
│ • Explains WHERE patterns are in pixels  │
└─────────────────────────────────────────┘
          ↓ SCALE BY ↓
┌─────────────────────────────────────────┐
│ Σ (784 values): IMPORTANCE WEIGHTS      │
│ • Large σᵢ = important pattern           │
│ • Small σᵢ = minor pattern/noise         │
│ • σ₁²/sum = {(sigma[0]**2/np.sum(sigma**2))*100:.1f}% (dominates!)      │
│ • Explains HOW MUCH of each pattern     │
└─────────────────────────────────────────┘
          ↓ MULTIPLY BY ↓
┌─────────────────────────────────────────┐
│ V^T (784 × 70000): COEFFICIENTS         │
│ • Recipe for each image                  │
│ • Each column = one image's coefficients │
│ • V^T_{i,j} = how much mode i in img j  │
│ • Explains WHICH patterns in each image │
└─────────────────────────────────────────┘

KEY INSIGHT:
Image = Weighted sum of basis patterns
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
        fontsize=9.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/q3_matrix_interpretation.png', 
            dpi=150, bbox_inches='tight')
print("    Figure saved as 'q3_matrix_interpretation.png'")
plt.close()

# ============ CONCRETE EXAMPLE ============
print("\n[7] Concrete Example with Real Image:")
print("    " + "-" * 66)

# Use pre-loaded data from SVD
A_orig = U @ np.diag(sigma) @ Vt

# Pick a sample image (digit 7)
test_digit = 7
test_indices = np.where(labels == test_digit)[0]
test_idx_actual = test_indices[0]

print(f"\nImage: Digit {test_digit} (first occurrence in dataset)")
print(f"Original image vector shape: (784,)")

# Reconstruct using different numbers of modes
print(f"\nReconstruction with different numbers of modes:")

modes_to_show = [1, 3, 5, 10, 20, 50, 102]
for r in modes_to_show:
    # Get original image coefficients
    image_coeffs_full = Vt[:, test_idx_actual]
    
    # Reconstruct with r modes
    U_r = U[:, :r]
    sigma_r = sigma[:r]
    Vt_r_col = Vt[: r, test_idx_actual]
    
    reconstructed = U_r @ np.diag(sigma_r) @ Vt_r_col
    
    # Get original from full reconstruction
    original = U @ np.diag(sigma) @ Vt[:, test_idx_actual]
    error = np.linalg.norm(original - reconstructed)
    
    print(f"  {r:3d} modes: reconstruction error = {error:.6f}")

print("\n" + "=" * 70)
print("QUESTION 3 SUMMARY:")
print("=" * 70)
print(f"""
U MATRIX (784 × 784):
✓ Contains left singular vectors (orthonormal basis)
✓ Each column = one pixel pattern/eigenface
✓ Columns u₁, u₂, ... are ordered by importance
✓ Interpretation: Spatial basis patterns common to all digits

Σ MATRIX (784 singular values):
✓ Diagonal matrix of singular values  
✓ σ₁ = {sigma[0]:.2f} (largest - most important)
✓ σᵢ represents importance/energy of mode i
✓ σ₁² alone accounts for {(sigma[0]**2/np.sum(sigma**2))*100:.1f}% of total variance
✓ Interpretation: Scaling factors showing importance of each pattern

V^T MATRIX (784 × 70000):
✓ Right singular vectors (image coefficients)
✓ Each column = one image in mode space
✓ V^T_{{i,j}} = coefficient of mode i in image j
✓ These ARE the PCA coordinates for classifiers!
✓ Interpretation: Recipe showing which patterns compose each image

COMBINED INTERPRETATION:
✓ A = U·Σ·V^T reconstructs original data perfectly
✓ U = "vocabulary" of pixel patterns
✓ Σ = "importance" of each pattern
✓ V^T = "recipe" for each image
✓ Low rank (~102 modes) explains 95% variance → compression!
""")
print("=" * 70)
