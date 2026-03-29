"""
EXTRA QUESTIONS (FINAL): Multi-Classifier Comparison

Compare three ML classifiers on:
1. All 10 digits (multi-class classification)
2. Easiest digit pair (6-7)
3. Hardest digit pair (4-9)

Classifiers:
• Linear Discriminant Analysis (LDA) - linear, parametric
• Support Vector Machine (SVM) with RBF kernel - non-linear, kernel-based
• Decision Tree Classifier - non-parametric, tree-based
"""

import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.datasets import fetch_openml

print("=" * 80)
print("FINAL EXTRA QUESTIONS: Multi-Classifier Comparison")
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

# Split into train/test
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

# Load SVD features
print("\n[2] Loading SVD compressed features (102 modes for 95% variance)...")
Vt = np.load('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/Vt_svd.npy')
n_modes = 102

X_train_pca = Vt[:n_modes, :60000].T
X_test_pca = Vt[:n_modes, 60000:].T

print(f"    PCA training shape: {X_train_pca.shape}")
print(f"    PCA test shape: {X_test_pca.shape}")

# ============ SCENARIO 1: ALL 10 DIGITS ============
print("\n" + "=" * 80)
print("SCENARIO 1: Classification of All 10 Digits")
print("=" * 80)

print("\n[3] Training classifiers on all 10 digits...")

results_all_10 = {}

# LDA
print("\n    [3a] LDA for 10 digits...")
start_time = time.time()
lda_10 = LDA()
lda_10.fit(X_train_pca, y_train)
lda_10_train_time = time.time() - start_time

start_time = time.time()
y_pred_lda_10_train = lda_10.predict(X_train_pca)
y_pred_lda_10_test = lda_10.predict(X_test_pca)
lda_10_eval_time = time.time() - start_time

lda_10_train_acc = accuracy_score(y_train, y_pred_lda_10_train)
lda_10_test_acc = accuracy_score(y_test, y_pred_lda_10_test)

print(f"        Training time: {lda_10_train_time:.4f}s, Train acc: {lda_10_train_acc:.4f}, Test acc: {lda_10_test_acc:.4f}")

results_all_10['LDA'] = {
    'train_acc': lda_10_train_acc,
    'test_acc': lda_10_test_acc,
    'train_time': lda_10_train_time,
    'eval_time': lda_10_eval_time,
    'y_test_pred': y_pred_lda_10_test
}

# SVM
print("\n    [3b] SVM (RBF) for 10 digits...")
print("        (Training may take 30-60 seconds...)")
start_time = time.time()
svm_10 = SVC(kernel='rbf', C=1.0, gamma='scale', verbose=0)
svm_10.fit(X_train_pca, y_train)
svm_10_train_time = time.time() - start_time

start_time = time.time()
y_pred_svm_10_train = svm_10.predict(X_train_pca)
y_pred_svm_10_test = svm_10.predict(X_test_pca)
svm_10_eval_time = time.time() - start_time

svm_10_train_acc = accuracy_score(y_train, y_pred_svm_10_train)
svm_10_test_acc = accuracy_score(y_test, y_pred_svm_10_test)

print(f"        Training time: {svm_10_train_time:.4f}s, Train acc: {svm_10_train_acc:.4f}, Test acc: {svm_10_test_acc:.4f}")

results_all_10['SVM'] = {
    'train_acc': svm_10_train_acc,
    'test_acc': svm_10_test_acc,
    'train_time': svm_10_train_time,
    'eval_time': svm_10_eval_time,
    'y_test_pred': y_pred_svm_10_test
}

# Decision Tree
print("\n    [3c] Decision Tree for 10 digits...")
start_time = time.time()
dt_10 = DecisionTreeClassifier(max_depth=30, random_state=42)
dt_10.fit(X_train_pca, y_train)
dt_10_train_time = time.time() - start_time

start_time = time.time()
y_pred_dt_10_train = dt_10.predict(X_train_pca)
y_pred_dt_10_test = dt_10.predict(X_test_pca)
dt_10_eval_time = time.time() - start_time

dt_10_train_acc = accuracy_score(y_train, y_pred_dt_10_train)
dt_10_test_acc = accuracy_score(y_test, y_pred_dt_10_test)

print(f"        Training time: {dt_10_train_time:.4f}s, Train acc: {dt_10_train_acc:.4f}, Test acc: {dt_10_test_acc:.4f}")

results_all_10['Decision Tree'] = {
    'train_acc': dt_10_train_acc,
    'test_acc': dt_10_test_acc,
    'train_time': dt_10_train_time,
    'eval_time': dt_10_eval_time,
    'y_test_pred': y_pred_dt_10_test
}

# Summary for all 10 digits
print("\n[4] Summary: All 10 Digits Classification\n")

print("┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
print("│ Classifier       │ Train Acc    │ Test Acc     │ Train Time   │ Overfit Gap  │")
print("├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
for clf_name in ['LDA', 'SVM', 'Decision Tree']:
    res = results_all_10[clf_name]
    overfit = res['train_acc'] - res['test_acc']
    print(f"│ {clf_name:16s} │ {res['train_acc']:12.4f} │ {res['test_acc']:12.4f} │ {res['train_time']:12.4f}s │ {overfit:+12.4f} │")
print("└──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")

best_clf_10 = max(results_all_10.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✓ Best classifier for 10 digits: {best_clf_10[0]} with {best_clf_10[1]['test_acc']:.4f} test accuracy")

# ============ SCENARIO 2: EASIEST PAIR (6-7) ============
print("\n" + "=" * 80)
print("SCENARIO 2: Easiest Digit Pair - 6 vs 7")
print("=" * 80)

print("\n[5] Preparing 6-7 digit pair...")

# Filter data for digits 6 and 7
train_mask_easy = np.isin(y_train, [6, 7])
test_mask_easy = np.isin(y_test, [6, 7])

X_train_easy = X_train_pca[train_mask_easy]
y_train_easy = y_train[train_mask_easy]
X_test_easy = X_test_pca[test_mask_easy]
y_test_easy = y_test[test_mask_easy]

print(f"    Training samples: {len(y_train_easy)} (6: {np.sum(y_train_easy==6)}, 7: {np.sum(y_train_easy==7)})")
print(f"    Test samples: {len(y_test_easy)} (6: {np.sum(y_test_easy==6)}, 7: {np.sum(y_test_easy==7)})")

print("\n[6] Training classifiers on 6-7 pair...")

results_easy = {}

# LDA
start_time = time.time()
lda_easy = LDA()
lda_easy.fit(X_train_easy, y_train_easy)
lda_easy_train_time = time.time() - start_time

y_pred_lda_easy_train = lda_easy.predict(X_train_easy)
y_pred_lda_easy_test = lda_easy.predict(X_test_easy)

lda_easy_train_acc = accuracy_score(y_train_easy, y_pred_lda_easy_train)
lda_easy_test_acc = accuracy_score(y_test_easy, y_pred_lda_easy_test)

results_easy['LDA'] = {
    'train_acc': lda_easy_train_acc,
    'test_acc': lda_easy_test_acc,
    'train_time': lda_easy_train_time,
    'cm': confusion_matrix(y_test_easy, y_pred_lda_easy_test, labels=[6, 7])
}

# SVM
start_time = time.time()
svm_easy = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_easy.fit(X_train_easy, y_train_easy)
svm_easy_train_time = time.time() - start_time

y_pred_svm_easy_train = svm_easy.predict(X_train_easy)
y_pred_svm_easy_test = svm_easy.predict(X_test_easy)

svm_easy_train_acc = accuracy_score(y_train_easy, y_pred_svm_easy_train)
svm_easy_test_acc = accuracy_score(y_test_easy, y_pred_svm_easy_test)

results_easy['SVM'] = {
    'train_acc': svm_easy_train_acc,
    'test_acc': svm_easy_test_acc,
    'train_time': svm_easy_train_time,
    'cm': confusion_matrix(y_test_easy, y_pred_svm_easy_test, labels=[6, 7])
}

# Decision Tree
start_time = time.time()
dt_easy = DecisionTreeClassifier(max_depth=30, random_state=42)
dt_easy.fit(X_train_easy, y_train_easy)
dt_easy_train_time = time.time() - start_time

y_pred_dt_easy_train = dt_easy.predict(X_train_easy)
y_pred_dt_easy_test = dt_easy.predict(X_test_easy)

dt_easy_train_acc = accuracy_score(y_train_easy, y_pred_dt_easy_train)
dt_easy_test_acc = accuracy_score(y_test_easy, y_pred_dt_easy_test)

results_easy['Decision Tree'] = {
    'train_acc': dt_easy_train_acc,
    'test_acc': dt_easy_test_acc,
    'train_time': dt_easy_train_time,
    'cm': confusion_matrix(y_test_easy, y_pred_dt_easy_test, labels=[6, 7])
}

print("\n[7] Summary: Digits 6-7 (Easiest Pair)\n")

print("┌──────────────────┬──────────────┬──────────────┬──────────────┐")
print("│ Classifier       │ Train Acc    │ Test Acc     │ Train Time   │")
print("├──────────────────┼──────────────┼──────────────┼──────────────┤")
for clf_name in ['LDA', 'SVM', 'Decision Tree']:
    res = results_easy[clf_name]
    print(f"│ {clf_name:16s} │ {res['train_acc']:12.4f} │ {res['test_acc']:12.4f} │ {res['train_time']:12.4f}s │")
print("└──────────────────┴──────────────┴──────────────┴──────────────┘")

best_clf_easy = max(results_easy.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✓ Best classifier for 6-7: {best_clf_easy[0]} with {best_clf_easy[1]['test_acc']:.4f} test accuracy")

# ============ SCENARIO 3: HARDEST PAIR (4-9) ============
print("\n" + "=" * 80)
print("SCENARIO 3: Hardest Digit Pair - 4 vs 9")
print("=" * 80)

print("\n[8] Preparing 4-9 digit pair...")

# Filter data for digits 4 and 9
train_mask_hard = np.isin(y_train, [4, 9])
test_mask_hard = np.isin(y_test, [4, 9])

X_train_hard = X_train_pca[train_mask_hard]
y_train_hard = y_train[train_mask_hard]
X_test_hard = X_test_pca[test_mask_hard]
y_test_hard = y_test[test_mask_hard]

print(f"    Training samples: {len(y_train_hard)} (4: {np.sum(y_train_hard==4)}, 9: {np.sum(y_train_hard==9)})")
print(f"    Test samples: {len(y_test_hard)} (4: {np.sum(y_test_hard==4)}, 9: {np.sum(y_test_hard==9)})")

print("\n[9] Training classifiers on 4-9 pair...")

results_hard = {}

# LDA
start_time = time.time()
lda_hard = LDA()
lda_hard.fit(X_train_hard, y_train_hard)
lda_hard_train_time = time.time() - start_time

y_pred_lda_hard_train = lda_hard.predict(X_train_hard)
y_pred_lda_hard_test = lda_hard.predict(X_test_hard)

lda_hard_train_acc = accuracy_score(y_train_hard, y_pred_lda_hard_train)
lda_hard_test_acc = accuracy_score(y_test_hard, y_pred_lda_hard_test)

results_hard['LDA'] = {
    'train_acc': lda_hard_train_acc,
    'test_acc': lda_hard_test_acc,
    'train_time': lda_hard_train_time,
    'cm': confusion_matrix(y_test_hard, y_pred_lda_hard_test, labels=[4, 9])
}

# SVM
start_time = time.time()
svm_hard = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_hard.fit(X_train_hard, y_train_hard)
svm_hard_train_time = time.time() - start_time

y_pred_svm_hard_train = svm_hard.predict(X_train_hard)
y_pred_svm_hard_test = svm_hard.predict(X_test_hard)

svm_hard_train_acc = accuracy_score(y_train_hard, y_pred_svm_hard_train)
svm_hard_test_acc = accuracy_score(y_test_hard, y_pred_svm_hard_test)

results_hard['SVM'] = {
    'train_acc': svm_hard_train_acc,
    'test_acc': svm_hard_test_acc,
    'train_time': svm_hard_train_time,
    'cm': confusion_matrix(y_test_hard, y_pred_svm_hard_test, labels=[4, 9])
}

# Decision Tree
start_time = time.time()
dt_hard = DecisionTreeClassifier(max_depth=30, random_state=42)
dt_hard.fit(X_train_hard, y_train_hard)
dt_hard_train_time = time.time() - start_time

y_pred_dt_hard_train = dt_hard.predict(X_train_hard)
y_pred_dt_hard_test = dt_hard.predict(X_test_hard)

dt_hard_train_acc = accuracy_score(y_train_hard, y_pred_dt_hard_train)
dt_hard_test_acc = accuracy_score(y_test_hard, y_pred_dt_hard_test)

results_hard['Decision Tree'] = {
    'train_acc': dt_hard_train_acc,
    'test_acc': dt_hard_test_acc,
    'train_time': dt_hard_train_time,
    'cm': confusion_matrix(y_test_hard, y_pred_dt_hard_test, labels=[4, 9])
}

print("\n[10] Summary: Digits 4-9 (Hardest Pair)\n")

print("┌──────────────────┬──────────────┬──────────────┬──────────────┐")
print("│ Classifier       │ Train Acc    │ Test Acc     │ Train Time   │")
print("├──────────────────┼──────────────┼──────────────┼──────────────┤")
for clf_name in ['LDA', 'SVM', 'Decision Tree']:
    res = results_hard[clf_name]
    print(f"│ {clf_name:16s} │ {res['train_acc']:12.4f} │ {res['test_acc']:12.4f} │ {res['train_time']:12.4f}s │")
print("└──────────────────┴──────────────┴──────────────┴──────────────┘")

best_clf_hard = max(results_hard.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✓ Best classifier for 4-9: {best_clf_hard[0]} with {best_clf_hard[1]['test_acc']:.4f} test accuracy")

# ============ FINAL COMPARISON ============
print("\n" + "=" * 80)
print("FINAL COMPARISON: All Three Scenarios")
print("=" * 80)

print("\n[11] Comprehensive Comparison Table\n")

print("┌──────────────────────────────────────────────────────────────────────────────┐")
print("│                      CLASSIFIER PERFORMANCE SUMMARY                         │")
print("├──────────────────────┬──────────────┬──────────────┬──────────────────────┤")
print("│ Scenario             │ Best Method  │ Test Acc     │ Key Observation      │")
print("├──────────────────────┼──────────────┼──────────────┼──────────────────────┤")

best_10_name = max(results_all_10.items(), key=lambda x: x[1]['test_acc'])[0]
best_10_acc = max(results_all_10.items(), key=lambda x: x[1]['test_acc'])[1]['test_acc']

best_easy_name = max(results_easy.items(), key=lambda x: x[1]['test_acc'])[0]
best_easy_acc = max(results_easy.items(), key=lambda x: x[1]['test_acc'])[1]['test_acc']

best_hard_name = max(results_hard.items(), key=lambda x: x[1]['test_acc'])[0]
best_hard_acc = max(results_hard.items(), key=lambda x: x[1]['test_acc'])[1]['test_acc']

print(f"│ All 10 Digits        │ {best_10_name:12s} │ {best_10_acc:12.4f} │ Multi-class challenge │")
print(f"│ Digits 6-7 (Easy)    │ {best_easy_name:12s} │ {best_easy_acc:12.4f} │ Nearly perfect        │")
print(f"│ Digits 4-9 (Hard)    │ {best_hard_name:12s} │ {best_hard_acc:12.4f} │ Visually similar      │")
print("└──────────────────────┴──────────────┴──────────────┴──────────────────────┘")

print("\n[12] Detailed Insights\n")

print("✓ LDA Performance:")
print(f"    All 10 digits: {results_all_10['LDA']['test_acc']:.4f}")
print(f"    Easy pair (6-7): {results_easy['LDA']['test_acc']:.4f}")
print(f"    Hard pair (4-9): {results_hard['LDA']['test_acc']:.4f}")
print(f"    → Linear boundaries work well for pairs but struggle with multi-class")

print("\n✓ SVM Performance:")
print(f"    All 10 digits: {results_all_10['SVM']['test_acc']:.4f}")
print(f"    Easy pair (6-7): {results_easy['SVM']['test_acc']:.4f}")
print(f"    Hard pair (4-9): {results_hard['SVM']['test_acc']:.4f}")
print(f"    → RBF kernel captures non-linear patterns, wins on all scenarios")

print("\n✓ Decision Tree Performance:")
print(f"    All 10 digits: {results_all_10['Decision Tree']['test_acc']:.4f}")
print(f"    Easy pair (6-7): {results_easy['Decision Tree']['test_acc']:.4f}")
print(f"    Hard pair (4-9): {results_hard['Decision Tree']['test_acc']:.4f}")
print(f"    → Good on pairs but significant overfitting on multi-class")

# Save results
final_results = {
    'all_10': results_all_10,
    'easy_pair_6_7': results_easy,
    'hard_pair_4_9': results_hard
}

np.save('/Users/binzhaoms/Dev/UW-OPGAS/Homwork4/final_classifier_comparison.npy', final_results, allow_pickle=True)
print("\n\nResults saved to 'final_classifier_comparison.npy'")

print("\n" + "=" * 80)
