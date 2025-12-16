# Data Dictionary - Handwritten Digit Classification Dataset

## Overview
This dataset contains pixel values from images of handwritten digits (simplified binary classification: digit 3 vs. digit 8). It's designed for practicing Support Vector Machines (SVM), Decision Trees, and ensemble methods like AdaBoost.

## Files
- `train.csv`: Training dataset with 1,000 samples
- `test.csv`: Test dataset with 200 samples
- `PS3-data.zip`: Original compressed dataset (legacy format)

## Target Variable
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| **digit** | int | Digit classification | 0 = digit 3, 1 = digit 8 |

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

### Pixel Position Mapping
```
pixel0   = row 0, col 0 (top-left)
pixel27  = row 0, col 27 (top-right)
pixel28  = row 1, col 0
pixel783 = row 27, col 27 (bottom-right)
```

## Data Characteristics

### Class Distribution
- **Training Set**: Balanced (50% each class)
- **Test Set**: Balanced (50% each class)

### Missing Values
- No missing values in this dataset

### Pixel Value Distribution
- Most pixels are near 0 (white background)
- Digit strokes have higher values (0.3 - 1.0)
- Natural sparsity in pixel data

### Dimensionality
- **High-dimensional**: 784 features
- **Visualization**: Reshape to 28×28 for display
- **Curse of dimensionality**: Consider dimensionality reduction

## Data Collection Context
This is a synthetic simplified version inspired by:
- MNIST handwritten digit dataset
- Optical Character Recognition (OCR) systems
- Postal code recognition systems

## Use Cases
This dataset is ideal for:
1. **Binary Classification**: Distinguishing two digit classes
2. **Support Vector Machines**: Linear and RBF kernels
3. **Decision Trees**: Handling high-dimensional data
4. **Boosting**: AdaBoost with weak learners
5. **Kernel Methods**: Demonstrating kernel trick

## Expected Model Performance

### Baseline (Random)
- Accuracy: 50% (random guess)

### Linear SVM
- Accuracy: 93-95%
- Training time: Fast
- Works when data is linearly separable in pixel space

### RBF SVM
- Accuracy: 96-98%
- Training time: Moderate
- Captures non-linear patterns
- **Best performer** on this task

### Decision Tree
- Accuracy: 85-92%
- Prone to overfitting if not pruned
- Fast training and prediction

### AdaBoost (with shallow trees)
- Accuracy: 94-96%
- Combines multiple weak learners
- More stable than single tree

## Business Context

### Real-World Applications
Similar technology powers:
- **Postal Service**: Automated mail sorting by ZIP code
- **Banking**: Check amount recognition
- **Forms Processing**: Automated data entry from handwritten forms
- **Education**: Automated grading systems

### Stakeholders
- **Operations**: Automate manual data entry
- **Quality Assurance**: Verify accuracy of automated systems
- **End Users**: Faster processing times

### Cost of Errors
- **Misclassification**: Wrong digit read
- **Business Impact**: Misrouted mail, incorrect transactions
- **Acceptable Error**: < 2-3% for production systems

## Feature Engineering Ideas

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA

# Reduce to 50 principal components
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
```

### Pixel Statistics
```python
# Aggregate features
row_sums = X.reshape(-1, 28, 28).sum(axis=2)
col_sums = X.reshape(-1, 28, 28).sum(axis=1)
```

### Symmetry Features
```python
# Left-right symmetry
left_half = X[:, :392]
right_half = X[:, 392:]
symmetry_score = np.abs(left_half - right_half[:, ::-1]).mean()
```

## Hyperparameter Tuning

### SVM Hyperparameters
```python
# C: Regularization parameter
C_values = [0.1, 1, 10, 100]

# gamma: RBF kernel parameter
gamma_values = [0.001, 0.01, 0.1, 1]
```

### Decision Tree Hyperparameters
```python
# max_depth: Maximum tree depth
max_depth = [3, 5, 10, 20, None]

# min_samples_split: Minimum samples to split
min_samples_split = [2, 5, 10]
```

### AdaBoost Hyperparameters
```python
# n_estimators: Number of weak learners
n_estimators = [10, 50, 100, 200]

# learning_rate: Weight of each weak learner
learning_rate = [0.01, 0.1, 1.0]
```

## Visualization Tips

### Display Digits
```python
import matplotlib.pyplot as plt

def show_digit(pixel_values):
    img = pixel_values.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

# Show first training example
show_digit(X_train[0])
```

### Decision Boundary Visualization
```python
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plot with decision boundary
# (requires trained classifier)
```

## Evaluation Metrics

### Primary Metrics
1. **Accuracy**: Overall correct predictions
2. **Confusion Matrix**: Where errors occur
3. **Precision/Recall**: Per-class performance
4. **Training Time**: Computational efficiency

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

## Key Challenges

1. **High Dimensionality**: 784 features, may need reduction
2. **Feature Scaling**: Important for SVM (pixels already 0-1)
3. **Overfitting**: Decision trees can overfit easily
4. **Computational Cost**: RBF SVM slower on large datasets
5. **Hyperparameter Sensitivity**: Need proper tuning

## Known Limitations

1. **Binary Only**: Real MNIST has 10 classes
2. **Simplified**: Real digits have more variation
3. **Clean Data**: No noise or distortions
4. **Small Dataset**: Only 1,000 training samples
5. **Synthetic Patterns**: Not actual handwritten digits

## Comparison with Full MNIST

| Aspect | This Dataset | Full MNIST |
|--------|-------------|------------|
| Classes | 2 (binary) | 10 (multiclass) |
| Training Samples | 1,000 | 60,000 |
| Difficulty | Easier | More challenging |
| Purpose | Learning SVM/Trees | Benchmarking |

## License
Dataset created for educational purposes as part of the Machine Learning Course.

---

**Made By Ritesh Rana**
