
Task (a): 100×100 Correlation Matrix (First 100 Images)
Input matrix slice: X_100 shape = (1024, 100)
Correlation matrix definition: C = X_100^T X_100

Key Results:
- Correlation matrix shape: (100, 100)
- Correlation matrix value range: [0.0004, 268.2986]
- Visualization saved as correlation_matrix_100x100.png

Task (b): Most Correlated and Most Uncorrelated Image Pairs (from Task a)
Computed from the 100×100 correlation matrix C (excluding diagonal self-correlations).

Key Results:
- Most highly correlated pair (1-based indices): (87, 89)
- Correlation value: 260.7754
- Plot saved as most_correlated_faces.png

- Most uncorrelated pair (1-based indices): (55, 65)
- Correlation value: 0.0022
- Plot saved as most_uncorrelated_faces.png

Addidonally, lookat the min of a and b. They are not the same, that is because: 
In Task (a), the reported minimum (0.0004) is from the full matrix (C), which includes diagonal entries (C_{ii} = x_i^T x_i) (self-correlation).
In your data, the global minimum is at ((65,65)): (0.00039594), which is a diagonal entry (same image with itself).
In Task (b), we intentionally exclude the diagonal to find correlation between two different images only.
The smallest off-diagonal value is (0.0021619) at pair ((55,65)), shown as 0.0022 after rounding.
So there’s no contradiction: (0.0004) is self-correlation; (0.0022) is the smallest cross-image correlation.

Task (c): 10×10 Correlation Matrix for Selected Images
Selected image IDs (1-based): [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
Correlation matrix definition: C_selected = X_selected^T X_selected

Key Results:
- Correlation matrix shape: (10, 10)
- Correlation matrix value range: [1.5454, 263.8802]
- Visualization saved as correlation_matrix_10x10_selected.png

Task (d): Eigenvectors of XX^T Matrix
Matrix Y = XX^T: Shape (1024, 1024), representing pixel-pixel correlations across all images
Eigenvalue Decomposition: Used np.linalg.eigh for symmetric matrix

Key Results:
First six eigenvalues (descending order):
- λ₁ = 234020.454854
- λ₂ = 49038.315301
- λ₃ = 8236.539897
- λ₄ = 6024.871458
- λ₅ = 2051.496433
- λ₆ = 1901.079115

First six eigenvectors (v₁ through v₆) computed and stored as 1024-dimensional vectors representing principal directions in the pixel space (matrix shape: (1024, 6)).
Saved output file: top6_eigenvectors_xxt.npy

Interpretation:
- The spectrum is strongly dominated by λ₁, with a clear drop to λ₂ and then to λ₃–λ₆.
- This indicates the face data has a low-dimensional structure in pixel space: a few directions explain most variation.
- The first mode likely captures global effects (overall face/illumination trend), while later modes capture finer variations.
- These results are consistent with Task (e), where λᵢ ≈ σᵢ² for the same principal directions.

Task (e): SVD of X and First Six Principal Component Directions
SVD performed as X = UΣV^T using np.linalg.svd(X, full_matrices=False)

Key Results:
- U shape: (1024, 1024)
- Singular values vector shape: (1024,)
- V^T shape: (1024, 2414)

First six singular values:
- σ₁ = 483.756607
- σ₂ = 221.445965
- σ₃ = 90.755385
- σ₄ = 77.620045
- σ₅ = 45.293448
- σ₆ = 43.601366

First six principal component directions (u₁ through u₆) extracted from the first six columns of U with shape (1024, 6).
Saved output file: top6_svd_principal_directions.npy

Task (f): Compare v₁ from (d) and u₁ from (e)
Computed norm of absolute-value difference:
|| |v₁| - |u₁| ||₂ = 0.000000000000

Interpretation:
- The near-zero difference confirms that the leading eigenvector of XX^T and the first left singular vector of X are equivalent up to sign.
- This is the expected theoretical result linking eigendecomposition of XX^T and SVD of X.

Saved output file: task_f_abs_diff_norm.npy

Task (g): Variance Captured by First 6 SVD Modes and Mode Plots
Variance percentage computed from singular values as:
variance_i (%) = (σ_i² / Σ_j σ_j²) × 100

Key Results:
- Mode 1: 72.927567%
- Mode 2: 15.281763%
- Mode 3: 2.566745%
- Mode 4: 1.877525%
- Mode 5: 0.639306%
- Mode 6: 0.592431%
- Cumulative variance (first 6 modes): 93.885337%

Saved output files:
- task_g_top6_variance_percent.npy
- task_g_variance_captured.png
- task_g_top6_svd_modes.png

Conclusion from Tasks (d)–(g)
- The eigendecomposition and SVD results both show that the Yale face data has strong low-dimensional structure.
- A few dominant modes capture most of the meaningful variation in the images.
- Task (f) gives || |v₁| - |u₁| ||₂ = 0, confirming that the leading eigenvector of XX^T and the first left singular vector of X are the same up to sign.
- Task (g) quantifies concentration of variance: the first six SVD modes capture 93.885337% total variance, with Mode 1 alone capturing 72.927567%.
- Overall, PCA/SVD is highly effective here for compact representation and analysis of face images.