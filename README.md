 
# 🧠 MNIST Neural Classifier

### Handwritten Digit Recognition Using PyTorch

![Python](https://img.shields.io/badge/Python-3.15-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![uv](https://img.shields.io/badge/Package%20Manager-uv-6A5ACD)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

A handwritten digit classification project built with **Python and PyTorch**, using a Multilayer Perceptron (MLP) trained on the MNIST dataset.

This project explores the complete foundational machine learning workflow: **data preprocessing, neural network architecture, training, validation, checkpointing, and evaluation on unseen data.**

---

## 📊 Model Performance

The trained model achieves **90.20% accuracy** on the MNIST test dataset.

| Metric | Result |
|---|---|
| 🎯 Test Accuracy | **90.20%** |
| ✅ Correct Predictions | 9,020 |
| ❌ Incorrect Predictions | 980 |
| 🧪 Test Samples | 10,000 |
| 🔁 Training Epochs | 3 |
| 🧠 Model Architecture | MLP |

> **Note:** This is a baseline neural network implementation intended for learning core deep learning concepts. Further evaluation and improvements are planned.

---

## 🏗️ Neural Network Architecture

The model uses a feedforward neural network with one hidden layer.

```text
             MNIST Image
              28 × 28
                 │
                 ▼
          ┌─────────────┐
          │   Flatten   │
          │  28×28 → 784│
          └─────────────┘
                 │
                 ▼
          ┌─────────────┐
          │ Linear Layer│
          │  784 → 128  │
          └─────────────┘
                 │
                 ▼
          ┌─────────────┐
          │    ReLU     │
          │ Activation  │
          └─────────────┘
                 │
                 ▼
          ┌─────────────┐
          │ Linear Layer│
          │   128 → 10  │
          └─────────────┘
                 │
                 ▼
          10 Output Scores
            Digits 0–9
```

**Architecture summary:**

- **Input:** 784 grayscale pixel values
- **Hidden layer:** 128 neurons
- **Activation:** ReLU
- **Output:** 10 class scores representing digits 0–9
- **Loss function:** Cross-Entropy Loss
- **Optimizer:** Stochastic Gradient Descent (SGD)

The predicted digit is selected using `argmax` over the output scores.

---

## 📁 Project Structure

```text
mnist-neural-classifier/
│
├── src/
│   └── mnist_neural_classifier/
│       ├── __init__.py
│       ├── model.py          # Neural network architecture
│       ├── train.py          # Training and validation
│       └── predict.py        # Test-set evaluation
│
├── scripts/
│   └── explore_data.py       # Dataset exploration
│
├── outputs/
│   └── metrics.json          # Training history
│
├── config.json               # Training configuration
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Locked dependencies
├── .python-version           # Python version
├── .gitignore
└── README.md
```

The training process also generates local files:

- `data/` — Downloaded MNIST dataset
- `checkpoints/best_model.pt` — Saved model weights

These generated files are excluded from Git tracking.

---

## 📦 Dataset

This project uses the **MNIST handwritten digit dataset**.

| Dataset Split | Images |
|---|---:|
| Training | 54,000 |
| Validation | 6,000 |
| Testing | 10,000 |
| **Total** | **70,000** |

Each image is a **28 × 28 grayscale image**, labeled with a digit from **0 to 9**.

The original 60,000 training images are divided into training and validation subsets using a fixed random seed (`42`).

**Dataset source:** [MNIST via Torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.15 (as configured in `.python-version`)
- [uv package manager](https://docs.astral.sh/uv/)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/maheenhmd/mnist-neural-classifier.git
cd mnist-neural-classifier
```

### 2. Install dependencies

```bash
uv sync
```

This installs the project dependencies, including PyTorch and Torchvision.

### 3. Explore the dataset

```bash
uv run python scripts/explore_data.py
```

This downloads MNIST if necessary and displays basic dataset information.

---

## 🚀 Train the Model

Run the training module from the project root:

```bash
uv run python -m mnist_neural_classifier.train
```

The training pipeline:

1. Downloads and loads MNIST.
2. Splits training data into training and validation subsets.
3. Processes images in batches.
4. Trains the MLP using SGD and Cross-Entropy Loss.
5. Evaluates performance after each epoch.
6. Saves the best-performing checkpoint based on validation loss.
7. Records training metrics.

### Training Configuration

```json
{
  "epochs": 3,
  "batch_size": 64,
  "learning_rate": 0.01
}
```

After training, the model checkpoint is saved to:

```text
checkpoints/best_model.pt
```

Training history is recorded in `outputs/metrics.json`.

---

## 🧪 Evaluate the Model

Once training is complete, run:

```bash
uv run python -m mnist_neural_classifier.predict
```

The evaluation script loads the saved model and measures performance on all **10,000 test images**.

### Example Output

```text
Total test images: 10000
Correct predictions: 9020
Incorrect predictions: 980
Test accuracy: 90.20%
```

The script uses `model.eval()` and `torch.no_grad()` to perform evaluation without updating model parameters.

Results may differ slightly when retraining the model.

---

## 📈 Training Results

Recorded training history from the repository:

| Epoch | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| 1 | 72.81% | 85.05% |
| 2 | 86.94% | 87.97% |
| 3 | 88.89% | 89.20% |

The model shows improvement across the three training epochs.

The final evaluation on the separate test dataset achieved **90.20% accuracy**.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming language |
| 🔥 PyTorch | Neural network training and inference |
| 🖼️ Torchvision | MNIST dataset and image preprocessing |
| 📦 uv | Dependency and environment management |
| 🔧 Git & GitHub | Version control and source hosting |

---

## 📚 Key Learning Outcomes

This project demonstrates practical experience with:

- Loading and preprocessing image datasets
- Building neural networks with `torch.nn`
- Understanding forward propagation
- Applying backpropagation and gradient descent
- Implementing training and validation loops
- Measuring model accuracy and loss
- Saving and loading trained model parameters
- Running batch inference on unseen test samples
- Organizing an ML project into reusable modules

---

## 🗺️ Future Improvements

- [ ] Generate a confusion matrix
- [ ] Visualize and analyze incorrect predictions
- [ ] Add automated unit tests
- [ ] Improve reproducibility
- [ ] Implement single-image inference through a dedicated interface
- [ ] Experiment with alternative model architectures

---

## 👩‍💻 Author

**Maheen Hamid**
Computer Science Graduate | AI/ML Engineer

[GitHub Profile](https://github.com/maheenhmd)

---

