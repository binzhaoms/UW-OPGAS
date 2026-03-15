import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load the Yale Faces dataset (expects yalefaces.mat in the same folder)
results = loadmat('yalefaces.mat')
X = results['X']


def plot_face_pair(X_data, idx1, idx2, title, output_file):
    face1 = X_data[:, idx1].reshape(32, 32, order='F')
    face2 = X_data[:, idx2].reshape(32, 32, order='F')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(face1, cmap='gray')
    axes[0].set_title(f'Image {idx1 + 1}')
    axes[0].axis('off')

    axes[1].imshow(face2, cmap='gray')
    axes[1].set_title(f'Image {idx2 + 1}')
    axes[1].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mode_grid(modes, title, output_file):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for mode_index, ax in enumerate(axes.flat):
        mode_image = modes[:, mode_index].reshape(32, 32, order='F')
        ax.imshow(mode_image, cmap='gray')
        ax.set_title(f'Mode {mode_index + 1}')
        ax.axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    # (a) Compute 100×100 correlation matrix for first 100 images
    print("\n(a) Computing 100×100 correlation matrix...")

    # Extract first 100 images (columns 0-99)
    X_100 = X[:, :100]
    print(f"Using first 100 images. Shape: {X_100.shape}")

    # Compute correlation matrix C where C[j,k] = x_j^T * x_k
    # This is equivalent to X_100.T @ X_100
    C = X_100.T @ X_100
    print(f"Correlation matrix shape: {C.shape}")
    print(f"Correlation matrix range: [{C.min():.4f}, {C.max():.4f}]")

    # Plot correlation matrix using pcolor
    plt.figure(figsize=(10, 8))
    plt.pcolor(C, cmap='viridis')
    plt.colorbar(label='Correlation')
    plt.title('100×100 Correlation Matrix (First 100 Face Images)')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.tight_layout()
    plt.savefig('correlation_matrix_100x100.png', dpi=300, bbox_inches='tight')
    print("Correlation matrix plot saved as 'correlation_matrix_100x100.png'")

    # (b) Find most highly correlated and most uncorrelated image pairs
    print("\n(b) Finding most highly correlated and most uncorrelated image pairs...")

    # Ignore self-correlation on the diagonal
    C_offdiag = C.copy()
    np.fill_diagonal(C_offdiag, -np.inf)
    most_corr_flat_idx = np.argmax(C_offdiag)
    most_corr_i, most_corr_j = np.unravel_index(most_corr_flat_idx, C_offdiag.shape)
    most_corr_value = C[most_corr_i, most_corr_j]

    # For uncorrelated pair, find minimum off-diagonal value
    C_offdiag_min = C.copy()
    np.fill_diagonal(C_offdiag_min, np.inf)
    most_uncorr_flat_idx = np.argmin(C_offdiag_min)
    most_uncorr_i, most_uncorr_j = np.unravel_index(most_uncorr_flat_idx, C_offdiag_min.shape)
    most_uncorr_value = C[most_uncorr_i, most_uncorr_j]

    print(
        f"Most highly correlated pair (1-based): "
        f"({most_corr_i + 1}, {most_corr_j + 1}) with value {most_corr_value:.4f}"
    )
    print(
        f"Most uncorrelated pair (1-based): "
        f"({most_uncorr_i + 1}, {most_uncorr_j + 1}) with value {most_uncorr_value:.4f}"
    )

    plot_face_pair(
        X_100,
        most_corr_i,
        most_corr_j,
        f'Most Correlated Faces (C={most_corr_value:.4f})',
        'most_correlated_faces.png'
    )
    print("Most correlated faces plot saved as 'most_correlated_faces.png'")

    plot_face_pair(
        X_100,
        most_uncorr_i,
        most_uncorr_j,
        f'Most Uncorrelated Faces (C={most_uncorr_value:.4f})',
        'most_uncorrelated_faces.png'
    )
    print("Most uncorrelated faces plot saved as 'most_uncorrelated_faces.png'")

    # (c) Compute 10×10 correlation matrix for selected images
    print("\n(c) Computing 10×10 correlation matrix for selected images...")
    selected_indices_1based = np.array([1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005])
    selected_indices_0based = selected_indices_1based - 1
    X_selected = X[:, selected_indices_0based]

    C_selected = X_selected.T @ X_selected
    print(f"Selected image IDs (1-based): {selected_indices_1based.tolist()}")
    print(f"Correlation matrix shape: {C_selected.shape}")
    print(f"Correlation matrix range: [{C_selected.min():.4f}, {C_selected.max():.4f}]")

    plt.figure(figsize=(10, 8))
    plt.pcolor(C_selected, cmap='viridis')
    plt.colorbar(label='Correlation')
    plt.title('10×10 Correlation Matrix (Selected Face Images)')
    plt.xlabel('Selected Image ID')
    plt.ylabel('Selected Image ID')
    tick_positions = np.arange(len(selected_indices_1based)) + 0.5
    tick_labels = selected_indices_1based.astype(str)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.yticks(tick_positions, tick_labels)
    plt.tight_layout()
    plt.savefig('correlation_matrix_10x10_selected.png', dpi=300, bbox_inches='tight')
    print("Correlation matrix plot saved as 'correlation_matrix_10x10_selected.png'")

    # (d) Compute first six eigenvectors of Y = X X^T
    print("\n(d) Computing first six eigenvectors of Y = X X^T...")
    Y = X @ X.T
    print(f"Y matrix shape: {Y.shape}")

    eigenvalues, eigenvectors = np.linalg.eigh(Y)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_desc = eigenvalues[sorted_indices]
    eigenvectors_desc = eigenvectors[:, sorted_indices]

    top6_eigenvalues = eigenvalues_desc[:6]
    top6_eigenvectors = eigenvectors_desc[:, :6]

    print("First six eigenvalues (descending):")
    for idx, val in enumerate(top6_eigenvalues, start=1):
        print(f"  λ{idx} = {val:.6f}")
    print(f"Top six eigenvectors shape: {top6_eigenvectors.shape}")

    np.save('top6_eigenvectors_xxt.npy', top6_eigenvectors)
    print("Top six eigenvectors saved as 'top6_eigenvectors_xxt.npy'")
    plot_mode_grid(top6_eigenvectors, 'Task (d): First 6 Eigenvectors of XX^T', 'task_d_top6_eigenvectors.png')
    print("Task (d) mode visualization saved as 'task_d_top6_eigenvectors.png'")

    # (e) SVD of X and first six principal component directions
    print("\n(e) Computing SVD of X and first six principal component directions...")
    U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    top6_pcs = U[:, :6]
    top6_singular_values = singular_values[:6]

    print(f"U shape: {U.shape}, singular values shape: {singular_values.shape}, Vt shape: {Vt.shape}")
    print("First six singular values:")
    for idx, val in enumerate(top6_singular_values, start=1):
        print(f"  σ{idx} = {val:.6f}")
    print(f"Top six principal directions shape: {top6_pcs.shape}")

    np.save('top6_svd_principal_directions.npy', top6_pcs)
    print("Top six principal directions saved as 'top6_svd_principal_directions.npy'")
    plot_mode_grid(top6_pcs, 'Task (e): First 6 SVD Principal Directions', 'task_e_top6_principal_directions.png')
    print("Task (e) mode visualization saved as 'task_e_top6_principal_directions.png'")

    # (f) Compare |v1| from (d) and |u1| from (e)
    print("\n(f) Computing norm difference between |v1| and |u1|...")
    v1 = top6_eigenvectors[:, 0]
    u1 = top6_pcs[:, 0]
    abs_diff_norm = np.linalg.norm(np.abs(v1) - np.abs(u1))
    print(f"|| |v1| - |u1| ||_2 = {abs_diff_norm:.12f}")

    np.save('task_f_abs_diff_norm.npy', np.array([abs_diff_norm]))
    print("Task (f) norm value saved as 'task_f_abs_diff_norm.npy'")

    # (g) Percentage variance captured by first 6 SVD modes and plot modes
    print("\n(g) Computing variance percentage captured by first 6 SVD modes...")
    singular_values_sq = singular_values ** 2
    total_variance = np.sum(singular_values_sq)
    top6_variance_pct = (singular_values_sq[:6] / total_variance) * 100

    print("Variance captured by first six SVD modes (%):")
    for idx, pct in enumerate(top6_variance_pct, start=1):
        print(f"  Mode {idx}: {pct:.6f}%")
    print(f"Cumulative variance (first 6 modes): {np.sum(top6_variance_pct):.6f}%")

    np.save('task_g_top6_variance_percent.npy', top6_variance_pct)
    print("Task (g) variance percentages saved as 'task_g_top6_variance_percent.npy'")

    plt.figure(figsize=(8, 5))
    mode_labels = [f'Mode {i}' for i in range(1, 7)]
    plt.bar(mode_labels, top6_variance_pct, color='teal')
    plt.ylabel('Variance Captured (%)')
    plt.title('Task (g): Variance Captured by First 6 SVD Modes')
    plt.tight_layout()
    plt.savefig('task_g_variance_captured.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Task (g) variance plot saved as 'task_g_variance_captured.png'")

    plot_mode_grid(top6_pcs, 'Task (g): First 6 SVD Modes', 'task_g_top6_svd_modes.png')
    print("Task (g) mode visualization saved as 'task_g_top6_svd_modes.png'")
