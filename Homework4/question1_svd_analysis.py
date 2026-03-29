"""
Question 1: SVD Analysis of MNIST Digit Images

Task: Perform Singular Value Decomposition (SVD) analysis on MNIST digit images.
Steps:
1. Load MNIST dataset
2. Reshape each image into a column vector
3. Create data matrix where each column is a different image
4. Perform SVD decomposition
"""

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

print("=" * 70)
print("QUESTION 1: SVD ANALYSIS OF MNIST DIGIT IMAGES")
print("=" * 70)

# Step 1: Load MNIST dataset
print("\n[1] Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

# Convert to numpy arrays and ensure proper data types
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=int)

print(f"    Dataset shape: {X.shape}")
print(f"    Number of images: {X.shape[0]}")
print(f"    Image dimensions: 28x28 = 784 pixels")
print(f"    Number of classes (digits 0-9): {len(np.unique(y))}")

# Step 2: Normalize pixel values to [0, 1]
print("\n[2] Normalizing pixel values...")
X = X / 255.0
print(f"    Pixel value range: [{X.min():.4f}, {X.max():.4f}]")

# Step 3: Create data matrix - each column is a reshaped image
print("\n[3] Creating data matrix...")
print("    Each column represents one image (784-dimensional vector)")
print("    Each row represents one pixel across all images")
# X is already in shape (70000, 784) - we need to transpose it
# so each column is an image
data_matrix = X.T  # Shape: (784, 70000)
print(f"    Data matrix shape: {data_matrix.shape}")

# Step 4: Perform Singular Value Decomposition (SVD)
print("\n[4] Performing Singular Value Decomposition (SVD)...")
print("    Computing U, Σ (Sigma), and V^T (V-transpose)...")
U, sigma, Vt = np.linalg.svd(data_matrix, full_matrices=False)

print(f"\n    U shape (left singular vectors): {U.shape}")
print(f"    Singular values (Σ) shape: {sigma.shape}")
print(f"    V^T shape (right singular vectors): {Vt.shape}")
print(f"    Effective rank: {len(sigma)}")

# Step 5: Display singular value statistics
print("\n[5] Singular Value Spectrum Analysis:")
print(f"    Largest singular value: {sigma[0]:.4f}")
print(f"    Smallest singular value: {sigma[-1]:.6f}")
print(f"    Condition number (σ_max/σ_min): {sigma[0]/sigma[-1]:.2e}")

# Step 6: Calculate cumulative explained variance
total_variance = np.sum(sigma**2)
cumsum_variance = np.cumsum(sigma**2) / total_variance

# Find number of modes needed for specific thresholds
threshold_90 = np.argmax(cumsum_variance >= 0.90) + 1
threshold_95 = np.argmax(cumsum_variance >= 0.95) + 1
threshold_99 = np.argmax(cumsum_variance >= 0.99) + 1

print(f"\n[6] Cumulative Explained Variance:")
print(f"    Modes needed for 90% variance: {threshold_90} out of {len(sigma)}")
print(f"    Modes needed for 95% variance: {threshold_95} out of {len(sigma)}")
print(f"    Modes needed for 99% variance: {threshold_99} out of {len(sigma)}")

# Step 7: Visualize singular value spectrum
print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Singular value spectrum (linear scale)
ax = axes[0, 0]
ax.plot(sigma[:100], 'b-', linewidth=2)
ax.set_xlabel('Mode Index')
ax.set_ylabel('Singular Value (Σ)')
ax.set_title('Singular Value Spectrum (First 100 modes)')
ax.grid(True, alpha=0.3)

# Plot 2: Singular value spectrum (log scale)
ax = axes[0, 1]
ax.semilogy(sigma, 'b-', linewidth=2)
ax.set_xlabel('Mode Index')
ax.set_ylabel('Singular Value (Σ) [log scale]')
ax.set_title('Singular Value Spectrum (Log Scale)')
ax.grid(True, alpha=0.3)

# Plot 3: Cumulative explained variance
ax = axes[1, 0]
ax.plot(cumsum_variance[:500], 'r-', linewidth=2, label='Cumulative Variance')
ax.axhline(y=0.90, color='g', linestyle='--', label='90% threshold')
ax.axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
ax.axhline(y=0.99, color='purple', linestyle='--', label='99% threshold')
ax.axvline(x=threshold_90, color='g', linestyle=':', alpha=0.5)
ax.axvline(x=threshold_95, color='orange', linestyle=':', alpha=0.5)
ax.axvline(x=threshold_99, color='purple', linestyle=':', alpha=0.5)
ax.set_xlabel('Number of Modes')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('Cumulative Explained Variance (First 500 modes)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: First 10 singular values
ax = axes[1, 1]
modes = np.arange(1, 11)
ax.bar(modes, sigma[:10], color='steelblue', edgecolor='black')
ax.set_xlabel('Mode Index')
ax.set_ylabel('Singular Value (Σ)')
ax.set_title('First 10 Singular Values')
ax.set_xticks(modes)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/Users/binzhaoms/Dev/UW-OPGAS/Homework4/q1_svd_spectrum.png', dpi=150, bbox_inches='tight')
print("    Figure saved as 'q1_svd_spectrum.png'")
plt.close()

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print(f"Data matrix: {data_matrix.shape[0]} pixels × {data_matrix.shape[1]} images")
print(f"U (left singular vectors): {U.shape[0]} × {U.shape[1]}")
print(f"  - Each column is a spatial basis mode (pixel space)")
print(f"Σ (singular values): {len(sigma)} values")
print(f"  - Measure the importance/energy of each mode")
print(f"V^T (right singular vectors): {Vt.shape[0]} × {Vt.shape[1]}")
print(f"  - Each row is image coefficients in the U basis")
print(f"\nEffective rank: {len(sigma)} (= min(784, 70000))")
print(f"For good reconstruction, approximately {threshold_95} modes ({threshold_95/len(sigma)*100:.1f}%) needed")
print("=" * 70)

# Save SVD components for later use
print("\n[8] Saving SVD components to files...")
np.save('/Users/binzhaoms/Dev/UW-OPGAS/Homework4/U_svd.npy', U)
np.save('/Users/binzhaoms/Dev/UW-OPGAS/Homework4/sigma_svd.npy', sigma)
np.save('/Users/binzhaoms/Dev/UW-OPGAS/Homework4/Vt_svd.npy', Vt)
np.save('/Users/binzhaoms/Dev/UW-OPGAS/Homework4/mnist_labels.npy', y)
print("    Saved: U_svd.npy, sigma_svd.npy, Vt_svd.npy, mnist_labels.npy")
