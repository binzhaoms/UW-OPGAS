"""
EXTRA QUESTIONS 3 & 4: LDA Analysis on All Digit Pairs

Build LDA classifiers for all possible pairs of digits (C(10,2) = 45 pairs)
and find which pairs are easiest and hardest to distinguish.

EXTRA QUESTION 3: Train LDA on all digit pairs and store accuracies
EXTRA QUESTION 4: Identify highest and lowest accuracy pairs, analyze results
"""

import numpy as np
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import fetch_openml

print("=" * 80)
print("EXTRA QUESTIONS 3 & 4: LDA Analysis on All Digit Pairs")
print("=" * 80)

# ============ LOAD DATA ============
print("\n[1] Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

# Normalize to [0, 1]
X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y, dtype=int)

print(f"    Total samples: {X.shape[0]}")
print(f"    Features: {X.shape[1]}")

# Split into train/test (60000 train, 10000 test - standard MNIST split)
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

# Load SVD for faster feature representation (102 modes = 95% variance)
print("\n[2] Loading SVD compressed features (102 modes for 95% variance)...")
Vt = np.load('/Users/binzhaoms/Dev/UW-OPGAS/Homework4/Vt_svd.npy')
n_modes = 102

X_train_pca = Vt[:n_modes, :60000].T   # (60000, 102)
X_test_pca = Vt[:n_modes, 60000:].T    # (10000, 102)

print(f"    PCA training shape: {X_train_pca.shape}")
print(f"    PCA test shape: {X_test_pca.shape}")

# ============ GENERATE ALL DIGIT PAIRS ============
print("\n[3] Generating all possible digit pairs...")

all_digits = np.arange(10)
digit_pairs = list(itertools.combinations(all_digits, 2))
num_pairs = len(digit_pairs)

print(f"    Total digit pairs: {num_pairs}")
print(f"    Sample pairs: {digit_pairs[:5]} ... {digit_pairs[-3:]}")

# ============ TRAIN LDA ON ALL PAIRS ============
print(f"\n[4] Training LDA classifiers on all {num_pairs} pairs...")
print("    (Progress: ", end="", flush=True)

results = []

for idx, (digit_a, digit_b) in enumerate(digit_pairs):
    # Filter training data
    train_mask = np.isin(y_train, [digit_a, digit_b])
    X_train_pair = X_train_pca[train_mask]
    y_train_pair = y_train[train_mask]
    
    # Filter test data
    test_mask = np.isin(y_test, [digit_a, digit_b])
    X_test_pair = X_test_pca[test_mask]
    y_test_pair = y_test[test_mask]
    
    # Train LDA
    lda = LDA()
    lda.fit(X_train_pair, y_train_pair)
    
    # Evaluate
    y_train_pred = lda.predict(X_train_pair)
    y_test_pred = lda.predict(X_test_pair)
    
    train_acc = accuracy_score(y_train_pair, y_train_pred)
    test_acc = accuracy_score(y_test_pair, y_test_pred)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test_pair, y_test_pred, labels=[digit_a, digit_b])
    
    # Store results
    results.append({
        'digit_a': digit_a,
        'digit_b': digit_b,
        'pair_str': f"{digit_a}-{digit_b}",
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_samples': len(y_train_pair),
        'test_samples': len(y_test_pair),
        'confusion_matrix': cm
    })
    
    # Progress indicator
    if (idx + 1) % 10 == 0:
        print(f"{idx + 1}", end=" ", flush=True)

print("✓)")

# ============ SORT AND ANALYZE RESULTS ============
print("\n[5] Analyzing results...")

# Sort by test accuracy
results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)

# Find best and worst pairs
best_pair = results_sorted[0]
worst_pair = results_sorted[-1]

print(f"\n    Best separable pair: Digits {best_pair['digit_a']} & {best_pair['digit_b']}")
print(f"    Worst separable pair: Digits {worst_pair['digit_a']} & {worst_pair['digit_b']}")

# ============ DISPLAY FULL RANKING ============
print("\n" + "=" * 80)
print("EXTRA QUESTION 3: ALL DIGIT PAIRS RANKED BY TEST ACCURACY")
print("=" * 80)

print("\n┌────┬──────────┬──────────┬──────────┬──────────┬──────────────┐")
print("│ #  │ Digit    │ Train    │ Test     │ Train N  │ Test N       │")
print("│    │ Pair     │ Accuracy │ Accuracy │ Samples  │ Samples      │")
print("├────┼──────────┼──────────┼──────────┼──────────┼──────────────┤")

for rank, result in enumerate(results_sorted, 1):
    digit_pair = f"{result['digit_a']}-{result['digit_b']}"
    print(f"│ {rank:2d} │ {digit_pair:8s} │ {result['train_acc']:.6f} │ {result['test_acc']:.6f} │ {result['train_samples']:8d} │ {result['test_samples']:12d} │")

print("└────┴──────────┴──────────┴──────────┴──────────┴──────────────┘")

# ============ EXTRA QUESTION 4: DETAILED ANALYSIS ============
print("\n" + "=" * 80)
print("EXTRA QUESTION 4: DETAILED ANALYSIS OF EXTREME PAIRS")
print("=" * 80)

# Top 5 best pairs
print("\n[6] TOP 5 EASIEST DIGIT PAIRS TO DISTINGUISH\n")

print("┌─────┬────────────┬──────────────┬──────────────┐")
print("│ #   │ Digit Pair │ Test Accuracy│ Overfitting  │")
print("├─────┼────────────┼──────────────┼──────────────┤")

for rank, result in enumerate(results_sorted[:5], 1):
    digit_pair = f"{result['digit_a']}-{result['digit_b']}"
    overfit = result['train_acc'] - result['test_acc']
    print(f"│ {rank}   │ {digit_pair:10s} │  {result['test_acc']:.6f}    │  {overfit:+.6f}    │")

print("└─────┴────────────┴──────────────┴──────────────┘")

# Bottom 5 worst pairs
print("\n[7] TOP 5 HARDEST DIGIT PAIRS TO DISTINGUISH\n")

print("┌─────┬────────────┬──────────────┬──────────────┐")
print("│ #   │ Digit Pair │ Test Accuracy│ Overfitting  │")
print("├─────┼────────────┼──────────────┼──────────────┤")

for rank, result in enumerate(results_sorted[-5:], 1):
    digit_pair = f"{result['digit_a']}-{result['digit_b']}"
    overfit = result['train_acc'] - result['test_acc']
    print(f"│ {5-rank+1}   │ {digit_pair:10s} │  {result['test_acc']:.6f}    │  {overfit:+.6f}    │")

print("└─────┴────────────┴──────────────┴──────────────┘")

# Detailed analysis of best pair
print("\n[8] DETAILED ANALYSIS: BEST PAIR")
print(f"\n    Digits: {best_pair['digit_a']} and {best_pair['digit_b']}")
print(f"    Training Accuracy: {best_pair['train_acc']:.6f}")
print(f"    Test Accuracy: {best_pair['test_acc']:.6f}")
print(f"    Training Samples: {best_pair['train_samples']}")
print(f"    Test Samples: {best_pair['test_samples']}")
print(f"\n    Confusion Matrix (Test Set):")
cm_best = best_pair['confusion_matrix']
print(f"                 Predicted")
print(f"                 {best_pair['digit_a']:3d}    {best_pair['digit_b']:3d}")
print(f"    Actual {best_pair['digit_a']}  {cm_best[0,0]:4d}   {cm_best[0,1]:4d}")
print(f"           {best_pair['digit_b']}  {cm_best[1,0]:4d}   {cm_best[1,1]:4d}")

recall_a = cm_best[0,0] / (cm_best[0,0] + cm_best[0,1])
recall_b = cm_best[1,1] / (cm_best[1,0] + cm_best[1,1])
print(f"    Recall {best_pair['digit_a']}: {recall_a:.6f}")
print(f"    Recall {best_pair['digit_b']}: {recall_b:.6f}")

# Detailed analysis of worst pair
print("\n[9] DETAILED ANALYSIS: WORST PAIR")
print(f"\n    Digits: {worst_pair['digit_a']} and {worst_pair['digit_b']}")
print(f"    Training Accuracy: {worst_pair['train_acc']:.6f}")
print(f"    Test Accuracy: {worst_pair['test_acc']:.6f}")
print(f"    Training Samples: {worst_pair['train_samples']}")
print(f"    Test Samples: {worst_pair['test_samples']}")
print(f"\n    Confusion Matrix (Test Set):")
cm_worst = worst_pair['confusion_matrix']
print(f"                 Predicted")
print(f"                 {worst_pair['digit_a']:3d}    {worst_pair['digit_b']:3d}")
print(f"    Actual {worst_pair['digit_a']}  {cm_worst[0,0]:4d}   {cm_worst[0,1]:4d}")
print(f"           {worst_pair['digit_b']}  {cm_worst[1,0]:4d}   {cm_worst[1,1]:4d}")

recall_a = cm_worst[0,0] / (cm_worst[0,0] + cm_worst[0,1])
recall_b = cm_worst[1,1] / (cm_worst[1,0] + cm_worst[1,1])
print(f"    Recall {worst_pair['digit_a']}: {recall_a:.6f}")
print(f"    Recall {worst_pair['digit_b']}: {recall_b:.6f}")

# Calculate accuracy statistics
accuracies = [r['test_acc'] for r in results]
avg_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
min_acc = np.min(accuracies)
max_acc = np.max(accuracies)

print("\n[10] SUMMARY STATISTICS")
print(f"\n    Average test accuracy across all pairs: {avg_acc:.6f}")
print(f"    Std Dev: {std_acc:.6f}")
print(f"    Min accuracy: {min_acc:.6f} (pair {worst_pair['digit_a']}-{worst_pair['digit_b']})")
print(f"    Max accuracy: {max_acc:.6f} (pair {best_pair['digit_a']}-{best_pair['digit_b']})")
print(f"    Range: {max_acc - min_acc:.6f}")

# Save results
results_dict = {
    'all_results': results_sorted,
    'best_pair': best_pair,
    'worst_pair': worst_pair,
    'avg_accuracy': avg_acc,
    'std_accuracy': std_acc,
    'min_accuracy': min_acc,
    'max_accuracy': max_acc
}

np.save('/Users/binzhaoms/Dev/UW-OPGAS/Homework4/extra_q3_q4_pairwise_results.npy', results_dict, allow_pickle=True)
print("\nResults saved to 'extra_q3_q4_pairwise_results.npy'")

print("\n" + "=" * 80)
