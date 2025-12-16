# Problem Set 3

## 1. Objective

This problem set covers advanced classification and ensemble methods:
- Implement Support Vector Machines (SVM) with different kernels
- Build decision trees from scratch
- Implement boosting algorithms (AdaBoost)
- Understand kernel trick and non-linear decision boundaries
- Compare linear vs non-linear models

You'll work on image classification, distinguishing handwritten digits.

## 2. Concepts Covered

- **Support_Vector_Machines/SVM**: Maximum margin classifiers
- **Support_Vector_Machines/Kernels**: RBF, polynomial kernels for non-linear boundaries
- **Tree_Based_Models/Decision_Trees**: Recursive splitting and entropy
- **Tree_Based_Models/Boosting**: AdaBoost ensemble method
- **Model_Evaluation**: Cross-validation and hyperparameter tuning

## 3. Dataset Description

**Dataset: Handwritten Digit Classification (subset of MNIST)**

Classify images of handwritten digits (0-9), simplified to binary (e.g., 3 vs 8).

**Features:**
- 784 features (28x28 pixel values)
- Each pixel value 0-255 (grayscale)
- Normalized to [0, 1]

**Target:**
- Binary classification (digit 3 vs digit 8)

**Real-world Analogy**: Like OCR systems that read checks, forms, or postal codes.

## 4. Tasks

### Task 1: Data Exploration and Visualization
- Load from `Data/PS3-data.zip`
- Visualize sample digits from each class
- Check pixel value distributions
- Dimensionality considerations

### Task 2: Implement Linear SVM
- Use sklearn's SVC with linear kernel
- Train on training set
- Evaluate accuracy and support vectors
- Visualize decision boundary (if possible with dimensionality reduction)

### Task 3: Implement SVM with RBF Kernel
- Use RBF (Gaussian) kernel
- Tune hyperparameters (C and gamma)
- Compare with linear SVM
- Analyze which is better for this data

### Task 4: Implement Decision Tree
- Build from scratch or use sklearn
- Visualize tree structure
- Analyze feature importance
- Check for overfitting (tree depth)

### Task 5: Implement AdaBoost
- Combine multiple weak learners (shallow trees)
- Implement weight update rule
- Track training error over iterations
- Compare with single decision tree

### Task 6: Model Comparison
- Compare all models on test set
- Training time vs accuracy tradeoff
- Hyperparameter sensitivity
- Interpretability vs performance

## 5. Expected Output

### Performance Benchmarks
- Linear SVM: ~95% accuracy
- RBF SVM: ~98% accuracy  
- Decision Tree: ~90% accuracy (single tree, prone to overfit)
- AdaBoost: ~96% accuracy

### Visualizations
- Sample digits from dataset
- Decision boundaries (with PCA/t-SNE)
- Learning curves
- Feature importance (for trees)

### Analysis
- Why does RBF kernel outperform linear?
- When would you use each model?
- Computational cost comparison

## 6. Interview Perspective

**Key Questions:**

1. **"Explain the kernel trick"**
   - Maps data to higher dimensions without explicit computation
   - Inner products in high-dim space computed efficiently
   - Enables non-linear boundaries with linear methods

2. **"What's the margin in SVM?"**
   - Distance from decision boundary to nearest points
   - SVM maximizes this margin
   - Larger margin → better generalization

3. **"How does boosting work?"**
   - Sequential ensemble: each model fixes previous errors
   - Reweight examples: focus on misclassified
   - Final prediction: weighted vote

4. **"Decision trees vs random forests?"**
   - Single tree: High variance, prone to overfitting
   - Random forest: Ensemble of trees, lower variance
   - RF averages multiple trees for better generalization

## 7. Common Mistakes

- Using default hyperparameters (C, gamma) without tuning
- Not scaling features for SVM (very important!)
- Growing trees too deep (overfit)
- Not using cross-validation for hyperparameter search
- Comparing models without considering computational cost

## 8. Extension Ideas (Optional)

- Implement multi-class classification (all 10 digits)
- Try other kernels (polynomial, sigmoid)
- Implement Random Forest
- Use PCA for dimensionality reduction first
- Build a real-time digit recognizer app

## 9. Data Files

From `Data/PS3-data.zip`:
- `digits_train.csv`: Training images and labels
- `digits_test.csv`: Test images and labels
- Each row: 784 pixel values + 1 label

## 10. Submission Checklist

- [ ] All models implemented and trained
- [ ] Hyperparameters tuned properly
- [ ] Performance comparison completed
- [ ] Visualizations included
- [ ] Understanding of kernel trick demonstrated
- [ ] Analysis of tradeoffs documented

**Time estimate**: 6-8 hours

---

**Made By Ritesh Rana**
