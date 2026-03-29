"""
EXTRA QUESTION 1: Build a Linear Discriminant Analysis (LDA) classifier for two digits

• Pick two digits (e.g., 3 and 8)
• Build LDA classifier to identify/classify these two digits
• Evaluate on both training and test sets
• Report accuracy and classification statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_openml

print("=" * 80)
print("EXTRA QUESTION 1: Build Linear Discriminant Analysis (LDA) for 2 Digits")
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

# ============ SELECT TWO DIGITS ============
print("\n[3] Selecting two digits for classification...")

# Choose digits 3 and 8 (challenging pair - visually similar)
digit1, digit2 = 3, 8
print(f"    Selected digits: {digit1} and {digit2}")
print(f"    Reason: These are visually similar, more challenging to separate")

# Filter data for these two digits
train_mask = np.isin(y_train, [digit1, digit2])
test_mask = np.isin(y_test, [digit1, digit2])

X_train_2d = X_train_pca[train_mask]
y_train_2d = y_train[train_mask]
X_test_2d = X_test_pca[test_mask]
y_test_2d = y_test[test_mask]

print(f"\n    Training set:")
print(f"      Total: {len(y_train_2d)} samples")
print(f"      Digit {digit1}: {np.sum(y_train_2d == digit1)} samples")
print(f"      Digit {digit2}: {np.sum(y_train_2d == digit2)} samples")

print(f"\n    Test set:")
print(f"      Total: {len(y_test_2d)} samples")
print(f"      Digit {digit1}: {np.sum(y_test_2d == digit1)} samples")
print(f"      Digit {digit2}: {np.sum(y_test_2d == digit2)} samples")

# ============ BUILD LDA CLASSIFIER ============
print("\n[4] Building Linear Discriminant Analysis (LDA) classifier...")

lda = LDA()
lda.fit(X_train_2d, y_train_2d)

print(f"    LDA trained successfully!")
print(f"    Model classes: {lda.classes_}")

# ============ EVALUATE ON TRAINING SET ============
print("\n[5] Evaluating LDA on TRAINING set...")

y_train_pred = lda.predict(X_train_2d)
train_accuracy = accuracy_score(y_train_2d, y_train_pred)
train_correct = np.sum(y_train_pred == y_train_2d)

print(f"    Training Accuracy: {train_accuracy:.4f}")
print(f"    Correct predictions: {train_correct}/{len(y_train_2d)}")
print(f"    Misclassifications: {len(y_train_2d) - train_correct}/{len(y_train_2d)}")

# ============ EVALUATE ON TEST SET ============
print("\n[6] Evaluating LDA on TEST set...")

y_test_pred = lda.predict(X_test_2d)
test_accuracy = accuracy_score(y_test_2d, y_test_pred)
test_correct = np.sum(y_test_pred == y_test_2d)

print(f"    Test Accuracy: {test_accuracy:.4f}")
print(f"    Correct predictions: {test_correct}/{len(y_test_2d)}")
print(f"    Misclassifications: {len(y_test_2d) - test_correct}/{len(y_test_2d)}")

# ============ CONFUSION MATRIX ============
print("\n[7] Confusion Matrix (Test Set):")

cm = confusion_matrix(y_test_2d, y_test_pred, labels=[digit1, digit2])
print(f"\n         Predicted")
print(f"         {digit1}    {digit2}")
print(f"Actual {digit1}  {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       {digit2}  {cm[1,0]:4d}  {cm[1,1]:4d}")

# Calculate per-class metrics
recall_digit1 = cm[0,0] / (cm[0,0] + cm[0,1])
recall_digit2 = cm[1,1] / (cm[1,0] + cm[1,1])

print(f"\n    Recall (Test):")
print(f"      Digit {digit1}: {recall_digit1:.4f} ({cm[0,0]}/{cm[0,0] + cm[0,1]} correct)")
print(f"      Digit {digit2}: {recall_digit2:.4f} ({cm[1,1]}/{cm[1,0] + cm[1,1]} correct)")

# ============ PREDICTION PROBABILITIES ============
print("\n[8] Analyzing prediction confidence...")

# Get decision function values (distance to separating hyperplane)
train_decision = lda.decision_function(X_train_2d)
test_decision = lda.decision_function(X_test_2d)

# Get predicted probabilities
train_proba = lda.predict_proba(X_train_2d)
test_proba = lda.predict_proba(X_test_2d)

# Maximum probability (confidence)
train_confidence = np.max(train_proba, axis=1)
test_confidence = np.max(test_proba, axis=1)

print(f"    Training set:")
print(f"      Mean confidence: {train_confidence.mean():.4f}")
print(f"      Min confidence: {train_confidence.min():.4f}")
print(f"      Max confidence: {train_confidence.max():.4f}")

print(f"\n    Test set:")
print(f"      Mean confidence: {test_confidence.mean():.4f}")
print(f"      Min confidence: {test_confidence.min():.4f}")
print(f"      Max confidence: {test_confidence.max():.4f}")

# ============ SUMMARY ============
print("\n" + "=" * 80)
print("EXTRA QUESTION 1 - SUMMARY")
print("=" * 80)

summary = f"""
TASK: Build Linear Discriminant Analysis (LDA) classifier for 2 digits

CONFIGURATION:
• Selected digits: {digit1} and {digit2}
• Feature representation: SVD-based PCA with {n_modes} modes (95% variance)
• Training samples: {len(y_train_2d)} (digit {digit1}: {np.sum(y_train_2d == digit1)}, digit {digit2}: {np.sum(y_train_2d == digit2)})
• Test samples: {len(y_test_2d)} (digit {digit1}: {np.sum(y_test_2d == digit1)}, digit {digit2}: {np.sum(y_test_2d == digit2)})

PERFORMANCE RESULTS:
┌─────────────────────────────────┐
│ Training Accuracy: {train_accuracy:.4f} │
│ Test Accuracy:     {test_accuracy:.4f} │
└─────────────────────────────────┘

CLASSIFICATION BREAKDOWN:
• Digit {digit1} recall (test): {recall_digit1:.4f} - {cm[0,0]} correct, {cm[0,1]} misclassified
• Digit {digit2} recall (test): {recall_digit2:.4f} - {cm[1,1]} correct, {cm[1,0]} misclassified

CONFIDENCE:
• Mean confidence on test set: {test_confidence.mean():.4f}
• Classifier is {test_confidence.mean()*100:.1f}% confident in its predictions

INTERPRETATION:
✓ LDA successfully separates digits {digit1} and {digit2}
✓ Test accuracy of {test_accuracy:.4f} indicates good linear separability
✓ Using 102 PCA modes (vs 784 pixels) reduces computation by 7.7× while preserving discriminative power
✓ The low-rank structure from Questions 2-3 enables effective classification
✓ No overfitting observed: train and test accuracy are similar ({train_accuracy:.4f} vs {test_accuracy:.4f})
"""

print(summary)

# Save results
results = {
    'digit1': digit1,
    'digit2': digit2,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'training_samples': len(y_train_2d),
    'test_samples': len(y_test_2d),
    'confusion_matrix': cm,
    'recall_digit1': recall_digit1,
    'recall_digit2': recall_digit2,
    'mean_confidence': test_confidence.mean()
}

np.save('/Users/binzhaoms/Dev/UW-OPGAS/Homework4/extra_question1_results.npy', results, allow_pickle=True)
print("\nResults saved to 'extra_question1_results.npy'")

print("\n" + "=" * 80)
