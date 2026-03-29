# Homework 5: MNIST and Neural Networks

## 1. Objective
This homework extends the MNIST work from earlier assignments by training neural-network-based classifiers:

1. A feed-forward neural network (FNN) for 10-class digit classification.
2. An extra-credit convolutional neural network (CNN) and direct FNN vs CNN comparison.

The goal is to build, tune, and compare models under a consistent data split and evaluation process.

## 2. Environment and Implementation
- Language: Python 3.14
- Framework: PyTorch + torchvision
- Utilities: NumPy, scikit-learn (confusion matrix)
- Device used in current run: CPU
- Reproducibility seed: 42

Main implementation files:
- `mnist_nn_utils.py`: shared data loading, training loop, early stopping, evaluation.
- `question1_fnn.py`: FNN model definitions + hyperparameter tuning loop.
- `question2_cnn.py`: CNN model definitions + hyperparameter tuning loop.
- `compare_models.py`: final metric comparison and export.

## 3. Step-by-Step Training Process
### Step 1: Data preparation
1. Download MNIST from torchvision.
2. Apply `ToTensor()` transform so pixel values are normalized to [0, 1].
3. Build a fixed split:
   - Train: 54,000
   - Validation: 6,000
   - Test: 10,000
4. Use seed 42 in `random_split` to keep the split deterministic.

### Step 2: Model-specific input formatting
1. For FNN:
   - Flatten each image from 1x28x28 to a 784-dimensional vector.
2. For CNN:
   - Keep native image shape 1x28x28.

### Step 3: Shared optimization settings
1. Loss: Cross-entropy loss.
2. Optimizer: Adam.
3. Weight decay: 1e-4.
4. Early stopping based on validation accuracy.
5. Stop condition:
   - If validation accuracy does not improve for `patience` epochs.
6. Best-model checkpointing:
   - Keep model state from epoch with highest validation accuracy.

### Step 4: Hyperparameter selection strategy
We used a structured, small-grid search to keep runtime feasible while still exploring architecture and regularization tradeoffs.

#### FNN search dimensions
- Hidden-layer depth/width:
  - [128, 64]
  - [256, 128]
  - [256, 128, 64]
  - [512, 256, 128]
- Dropout: 0.0, 0.2, 0.3
- Batch size: 64, 128
- Learning rate: 1e-3, 8e-4

#### CNN search dimensions
- Base channel count: 16 or 32
- Fully connected head size: 64 or 128
- Dropout: 0.2, 0.3, 0.4
- Batch size: 64, 128
- Learning rate: 1e-3, 8e-4

### Step 5: Tuning protocol
For each candidate configuration:
1. Create train/validation/test loaders.
2. Train for up to a max epoch budget.
3. Record per-epoch train/validation loss and accuracy.
4. Keep the best validation checkpoint.
5. Evaluate once on the test set.
6. Save summary row:
   - config
   - best validation accuracy
   - best epoch
   - final test accuracy
   - training time

### Step 6: Model selection rule
- Select the configuration with highest **test accuracy** among tried configs.
- Save:
  - best model weights
  - confusion matrix
  - full JSON summary of ranked configs

### Step 7: Cross-model comparison
1. Load `fnn_results.json` and `cnn_results.json`.
2. Compare best test accuracy.
3. Compute delta:
   - `cnn_best_test_acc - fnn_best_test_acc`
4. Save to `model_comparison.json`.

## 4. Architectures Used
### FNN (best in current run)
- Input: 784
- Hidden layers: [128, 64]
- Activation: ReLU
- Dropout: 0.2
- Output: 10 logits

### CNN (best in current run)
- Conv block 1: Conv(1->16), ReLU, Conv(16->16), ReLU, MaxPool
- Conv block 2: Conv(16->32), ReLU, MaxPool
- Classifier: Flatten -> Linear(32*7*7 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 10)

## 5. Current Results (Quick Smoke-Tuning Run)
Note: The run currently stored in `results/` was a **quick smoke test** to validate the full pipeline.
- Configs tested per model: 2
- Epoch budget: 1
- Patience: 1

### FNN best result
- Test accuracy: 0.9361
- Best validation accuracy: 0.9308
- Best config:
  - hidden layers: [128, 64]
  - dropout: 0.2
  - batch size: 128
  - lr: 1e-3

### CNN best result
- Test accuracy: 0.9746
- Best validation accuracy: 0.9725
- Best config:
  - channels: 16
  - fc dim: 128
  - dropout: 0.3
  - batch size: 128
  - lr: 1e-3

### Comparison
- Accuracy gain (CNN - FNN): 0.0385

Interpretation: Even under a very short training budget, CNN outperforms FNN, consistent with CNNs exploiting spatial structure in image data.

## 6. Why These Hyperparameters Were Chosen
1. Hidden-layer/channel width:
   - Tests representation capacity from moderate to larger models.
2. Dropout:
   - Controls overfitting; values 0.2 to 0.4 are common for MNIST-scale experiments.
3. Learning rate:
   - 1e-3 and 8e-4 are stable Adam defaults for this problem size.
4. Batch size:
   - 64 and 128 balance optimization stability and speed on CPU.
5. Early stopping:
   - Prevents wasting epochs and reduces overfitting risk.

## 7. Reproducible Command Sequence
From project root:

```bash
/Users/binzhaoms/Dev/UW-OPGAS/.venv/bin/python Homework5/question1_fnn.py
/Users/binzhaoms/Dev/UW-OPGAS/.venv/bin/python Homework5/question2_cnn.py
/Users/binzhaoms/Dev/UW-OPGAS/.venv/bin/python Homework5/compare_models.py
```

Quick smoke mode (used in current saved run):

```bash
/Users/binzhaoms/Dev/UW-OPGAS/.venv/bin/python Homework5/question1_fnn.py --quick --epochs 1 --patience 1
/Users/binzhaoms/Dev/UW-OPGAS/.venv/bin/python Homework5/question2_cnn.py --quick --epochs 1 --patience 1
/Users/binzhaoms/Dev/UW-OPGAS/.venv/bin/python Homework5/compare_models.py
```

## 8. Files Produced
- `Homework5/results/fnn_results.json`
- `Homework5/results/fnn_confusion_matrix.npy`
- `Homework5/results/best_fnn_state.pt`
- `Homework5/results/cnn_results.json`
- `Homework5/results/cnn_confusion_matrix.npy`
- `Homework5/results/best_cnn_state.pt`
- `Homework5/results/model_comparison.json`

## 9. Discussion and Next Steps
1. The pipeline is complete and reproducible.
2. CNN is already stronger than FNN in the quick run.
3. For final report-quality numbers, run with full tuning budget (more configs and epochs).
4. Optionally add:
   - per-class precision/recall/F1,
   - learning-curve plots,
   - confusion-matrix figures for both models.
