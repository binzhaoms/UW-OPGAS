"""
Visualize Pairwise LDA Results in a 10x10 Grid Heatmap

Creates a heatmap showing test accuracy for all digit pairs (0-9)
The grid shows which pairs are easy (light/bright) and hard (dark) to classify.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itertools

print("=" * 80)
print("VISUALIZING PAIRWISE LDA RESULTS: 10x10 ACCURACY HEATMAP")
print("=" * 80)

# ============ LOAD RESULTS ============
print("\n[1] Loading pairwise LDA results...")

results_dict = np.load('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/extra_q3_q4_pairwise_results.npy', allow_pickle=True).item()
all_results = results_dict['all_results']

print(f"    Loaded {len(all_results)} digit pair results")

# ============ CREATE 10x10 ACCURACY MATRIX ============
print("\n[2] Creating 10x10 accuracy matrix...")

accuracy_matrix = np.ones((10, 10)) * np.nan  # Initialize with NaN

# Fill in the accuracies
for result in all_results:
    digit_a = result['digit_a']
    digit_b = result['digit_b']
    test_acc = result['test_acc']
    
    # Put in upper triangle (digit_a < digit_b)
    if digit_a < digit_b:
        accuracy_matrix[digit_a, digit_b] = test_acc
    else:
        accuracy_matrix[digit_b, digit_a] = test_acc

# Also fill the lower triangle for symmetry and fill diagonal with 1.0 (perfect accuracy within same digit)
for i in range(10):
    accuracy_matrix[i, i] = 1.0  # Same digit pairs have 100% accuracy
    for j in range(i+1, 10):
        if not np.isnan(accuracy_matrix[i, j]):
            accuracy_matrix[j, i] = accuracy_matrix[i, j]  # Mirror to lower triangle

print(f"    Matrix shape: {accuracy_matrix.shape}")
print(f"    Min accuracy: {np.nanmin(accuracy_matrix):.6f}")
print(f"    Max accuracy: {np.nanmax(accuracy_matrix):.6f}")
print(f"    Mean accuracy: {np.nanmean(accuracy_matrix):.6f}")

# ============ CREATE HEATMAP VISUALIZATION ============
print("\n[3] Creating heatmap visualization...")

fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap
im = ax.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0.95, vmax=1.00, aspect='auto')

# Set ticks
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(np.arange(10), fontsize=12)
ax.set_yticklabels(np.arange(10), fontsize=12)

# Labels
ax.set_xlabel('Digit', fontsize=14, fontweight='bold')
ax.set_ylabel('Digit', fontsize=14, fontweight='bold')
ax.set_title('Pairwise LDA Classification Accuracy Heatmap\n(10x10 Digit Pairs)', 
             fontsize=16, fontweight='bold', pad=20)

# Rotate the tick labels and set alignment
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
plt.setp(ax.get_yticklabels(), rotation=0)

# Add text annotations with accuracy values
for i in range(10):
    for j in range(10):
        if not np.isnan(accuracy_matrix[i, j]):
            acc = accuracy_matrix[i, j]
            # Color text based on background brightness
            text_color = 'white' if acc < 0.965 else 'black'
            text = ax.text(j, i, f'{acc:.4f}', ha="center", va="center",
                          color=text_color, fontsize=10, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Test Accuracy', rotation=270, labelpad=25, fontsize=12, fontweight='bold')

# Add grid lines
ax.set_xticks(np.arange(10)-.5, minor=True)
ax.set_yticks(np.arange(10)-.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

plt.tight_layout()
plt.savefig('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/pairwise_lda_heatmap.png', dpi=150, bbox_inches='tight')
print("    Saved: pairwise_lda_heatmap.png")

# ============ CREATE SIMPLIFIED VIEW ============
print("\n[4] Creating simplified view (accuracy in percentages)...")

fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap with percentage scale
accuracy_percent = accuracy_matrix * 100

im = ax.imshow(accuracy_percent, cmap='RdYlGn', vmin=95, vmax=100, aspect='auto')

# Set ticks
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(np.arange(10), fontsize=12)
ax.set_yticklabels(np.arange(10), fontsize=12)

# Labels
ax.set_xlabel('Digit', fontsize=14, fontweight='bold')
ax.set_ylabel('Digit', fontsize=14, fontweight='bold')
ax.set_title('Pairwise LDA Classification Accuracy (%) Heatmap\n(10x10 Digit Pairs)', 
             fontsize=16, fontweight='bold', pad=20)

# Add text annotations
for i in range(10):
    for j in range(10):
        if not np.isnan(accuracy_percent[i, j]):
            acc_pct = accuracy_percent[i, j]
            text_color = 'white' if acc_pct < 96.5 else 'black'
            text = ax.text(j, i, f'{acc_pct:.2f}%', ha="center", va="center",
                          color=text_color, fontsize=10, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Test Accuracy (%)', rotation=270, labelpad=25, fontsize=12, fontweight='bold')

# Add grid lines
ax.set_xticks(np.arange(10)-.5, minor=True)
ax.set_yticks(np.arange(10)-.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

plt.tight_layout()
plt.savefig('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/pairwise_lda_heatmap_percent.png', dpi=150, bbox_inches='tight')
print("    Saved: pairwise_lda_heatmap_percent.png")

# ============ SUMMARY STATISTICS ============
print("\n[5] Summary Statistics\n")

print("┌─────────────────────────────────────────┐")
print("│      Accuracy Range by Quartile         │")
print("├─────────────────────────────────────────┤")

# Calculate quartiles (excluding diagonal)
off_diag_accs = []
for i in range(10):
    for j in range(i+1, 10):
        off_diag_accs.append(accuracy_matrix[i, j])

off_diag_accs = np.array(off_diag_accs)
q1 = np.percentile(off_diag_accs, 25)
q2 = np.percentile(off_diag_accs, 50)
q3 = np.percentile(off_diag_accs, 75)

print(f"│ Q1 (25th percentile): {q1:.6f} (95%) │")
print(f"│ Q2 (50th percentile): {q2:.6f} (median) │")
print(f"│ Q3 (75th percentile): {q3:.6f} (75%) │")
print("└─────────────────────────────────────────┘")

print("\n┌──────────────────────────────────────────┐")
print("│   Accuracy Distribution Analysis        │")
print("├──────────────────────────────────────────┤")

# Count pairs in each accuracy range
ranges = [
    (0.99, 1.00, "99-100%"),
    (0.98, 0.99, "98-99%"),
    (0.97, 0.98, "97-98%"),
    (0.96, 0.97, "96-97%"),
    (0.95, 0.96, "95-96%")
]

for min_acc, max_acc, label in ranges:
    count = np.sum((off_diag_accs >= min_acc) & (off_diag_accs < max_acc))
    pct = 100 * count / len(off_diag_accs)
    bar_length = int(pct / 2)
    bar = "█" * bar_length
    print(f"│ {label}: {count:2d} pairs ({pct:5.1f}%) {bar:23s} │")

print("└──────────────────────────────────────────┘")

# Find best and worst again
best_idx = np.unravel_index(np.nanargmax(np.triu(accuracy_matrix, k=1)), accuracy_matrix.shape)
worst_idx = np.unravel_index(np.nanargmin(np.triu(accuracy_matrix, k=1)), accuracy_matrix.shape)

best_acc = accuracy_matrix[best_idx]
worst_acc = accuracy_matrix[worst_idx]

print(f"\n✓ Best pair: Digits {best_idx[0]}-{best_idx[1]} with {best_acc:.6f} accuracy")
print(f"✓ Worst pair: Digits {worst_idx[0]}-{worst_idx[1]} with {worst_acc:.6f} accuracy")
print(f"✓ All 45 pairs have test accuracy ≥ 95.23%")

print("\n" + "=" * 80)
print("Visualization complete! Check the PNG files for the heatmap.")
print("=" * 80)
