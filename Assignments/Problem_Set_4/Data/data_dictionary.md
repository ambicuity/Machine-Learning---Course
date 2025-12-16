# Data Dictionary - MNIST Digit Classification (Full 10-Class) Dataset

## Overview
This dataset contains pixel values from images of handwritten digits (0-9) for multiclass classification. It's designed for practicing Neural Networks, Convolutional Neural Networks (CNNs), and comparing with classical ML methods.

## Files
- `train.csv`: Training dataset with 2,000 samples  
- `test.csv`: Test dataset with 400 samples
- `PS4-data.zip`: Original compressed dataset (legacy format)

## Target Variable
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| **digit** | int | Digit classification | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |

## Features

### Pixel Features (784 features)
Each image is 28x28 pixels, flattened into a single row:

| Feature Pattern | Description | Range |
|----------------|-------------|-------|
| pixel0 to pixel783 | Grayscale pixel intensity values | 0.0 - 1.0 (normalized) |

**Image Structure**:
- Original: 28 rows × 28 columns = 784 pixels
- Flattened: Single row with 784 values
- Pixel ordering: Row-major (left-to-right, top-to-bottom)

## Data Characteristics

### Class Distribution
- **Balanced**: 200 samples per digit class in training set
- **Test Set**: 40 samples per digit class (balanced)

### Missing Values
- No missing values in this dataset

### Pixel Value Distribution
- Most pixels are near 0 (white background)
- Digit strokes have higher values (0.2 - 1.0)
- Each digit class has distinctive pixel patterns

### Dimensionality
- **High-dimensional**: 784 input features
- **10 Output Classes**: Multiclass classification
- **Suitable for deep learning**: Complex non-linear patterns

## Data Collection Context
This is a simplified synthetic version inspired by:
- **MNIST**: The famous handwritten digit dataset
- **Real MNIST**: 60,000 training + 10,000 test samples
- Used as benchmark for computer vision algorithms

## Use Cases
This dataset is ideal for:
1. **Multiclass Classification**: 10 digit classes (0-9)
2. **Feedforward Neural Networks**: From scratch implementation
3. **Backpropagation**: Gradient computation and training
4. **Convolutional Neural Networks**: Image-specific architecture
5. **Comparison**: Deep learning vs classical ML methods

## Expected Model Performance

### Baseline (Random)
- Accuracy: 10% (random guess among 10 classes)

### Logistic Regression (Multiclass)
- Accuracy: 85-90%
- Fast but limited by linear decision boundaries

### Feedforward Neural Network (2 layers)
- Architecture: 784 → 128 → 10
- Accuracy: 95-97%
- ReLU activation, softmax output
- **Good starting point**

### Convolutional Neural Network
- Architecture: Conv → Pool → Conv → Pool → FC
- Accuracy: 98-99%
- Captures spatial features
- **Best performer**

### Real MNIST Benchmarks
- Simple NN: 96-98%
- CNN (LeNet-5): 98.5-99.2%
- Deep CNN (ResNet): 99.5%+

## Business Context

### Real-World Applications
- **Banking**: Check processing, amount recognition
- **Postal Services**: ZIP code reading for mail sorting
- **Mobile Apps**: Handwriting recognition keyboards
- **Education**: Automated grading and assessment
- **Healthcare**: Medical form digitization

### Stakeholders
- **Data Scientists**: Building and deploying models
- **Engineers**: Integrating models into production systems
- **End Users**: Benefiting from automated digit recognition
- **Business**: Cost savings from automation

### Cost of Errors
- **Misclassification**: Wrong digit recognized
- **Impact**: Financial errors, misrouted mail, bad UX
- **Production Standard**: > 98% accuracy required

## Neural Network Architecture Design

### Feedforward Network
```python
# Simple architecture
Layer 1: Dense(784 → 128, activation='relu')
Layer 2: Dense(128 → 10, activation='softmax')

# Loss: Categorical crossentropy
# Optimizer: SGD or Adam
# Metrics: Accuracy
```

### Convolutional Network
```python
# CNN architecture
Layer 1: Conv2D(32 filters, 3×3, activation='relu')
Layer 2: MaxPooling2D(2×2)
Layer 3: Conv2D(64 filters, 3×3, activation='relu')
Layer 4: MaxPooling2D(2×2)
Layer 5: Flatten()
Layer 6: Dense(128, activation='relu')
Layer 7: Dense(10, activation='softmax')
```

## Training Strategies

### Data Preprocessing
```python
# Normalize pixel values
X = X / 255.0  # Scale to [0, 1]

# Or standardize
X = (X - X.mean()) / X.std()

# Reshape for CNN
X_cnn = X.reshape(-1, 28, 28, 1)
```

### Train/Validation Split
```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Training Configuration
```python
# Hyperparameters
batch_size = 32
epochs = 10-20
learning_rate = 0.001

# Early stopping
monitor = 'val_loss'
patience = 3
```

## Backpropagation Implementation

### Forward Pass
```python
# Layer 1
z1 = X @ W1 + b1
a1 = relu(z1)

# Layer 2 (output)
z2 = a1 @ W2 + b2
a2 = softmax(z2)
```

### Backward Pass
```python
# Output layer gradient
dz2 = a2 - y_onehot

# Hidden layer gradients
dW2 = a1.T @ dz2
db2 = dz2.sum(axis=0)
da1 = dz2 @ W2.T
dz1 = da1 * (z1 > 0)  # ReLU derivative

# Input layer gradients
dW1 = X.T @ dz1
db1 = dz1.sum(axis=0)
```

## Feature Engineering Ideas

### Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
```

### Feature Extraction
```python
# Digit statistics
pixel_mean = X.mean(axis=1)
pixel_std = X.std(axis=1)
pixel_max = X.max(axis=1)

# Spatial features
center_mass_x = (X.reshape(-1, 28, 28) * np.arange(28)).sum() / X.sum()
center_mass_y = (X.reshape(-1, 28, 28) * np.arange(28)[:, None]).sum() / X.sum()
```

## Visualization

### Display Sample Digits
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.show()
```

### Training Curves
```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Evaluation Metrics

### Primary Metrics
1. **Accuracy**: Overall correct predictions
2. **Loss**: Cross-entropy loss value
3. **Per-Class Accuracy**: Performance on each digit
4. **Confusion Matrix**: Misclassification patterns

### Advanced Metrics
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
# Shows precision, recall, f1-score per class
```

## Key Challenges

1. **Vanishing Gradients**: Deep networks may struggle
   - Solution: ReLU activation, batch normalization
2. **Overfitting**: Model memorizes training data
   - Solution: Dropout, regularization, data augmentation
3. **Computational Cost**: Training can be slow
   - Solution: GPU acceleration, batch processing
4. **Hyperparameter Tuning**: Many parameters to optimize
   - Solution: Grid search, random search, Bayesian optimization
5. **Class Confusion**: Some digits look similar (e.g., 3 vs 8, 4 vs 9)

## Common Mistakes

### Implementation Mistakes
- Not normalizing pixel values
- Wrong loss function (use categorical crossentropy)
- Not using one-hot encoding for labels
- Learning rate too high (divergence)
- Not shuffling training data

### Architecture Mistakes
- Too few hidden units (underfitting)
- Too many layers without regularization (overfitting)
- Wrong activation (sigmoid instead of ReLU)
- No activation in hidden layers (linear model)

## Known Limitations

1. **Simplified Dataset**: Smaller than real MNIST
2. **No Noise**: Real handwriting has more variation
3. **Balanced Classes**: Real data often imbalanced
4. **Clean Images**: No distortions or artifacts
5. **Synthetic Patterns**: Not actual handwritten digits

## Comparison with Real MNIST

| Aspect | This Dataset | Real MNIST |
|--------|-------------|------------|
| Training Samples | 2,000 | 60,000 |
| Test Samples | 400 | 10,000 |
| Difficulty | Easier | Standard benchmark |
| Training Time | Minutes | Hours (without GPU) |
| Purpose | Learning | Research benchmark |

## Extensions

### Advanced Architectures
- Residual connections (ResNet)
- Batch normalization
- Dropout layers
- Skip connections

### Transfer Learning
- Use pretrained models
- Fine-tune on this dataset
- Feature extraction approach

## License
Dataset created for educational purposes as part of the Machine Learning Course.

---

**Made By Ritesh Rana**
