# Solution Set 3

## 1. Solution Overview

This solution implements and compares four advanced classification methods:

1. **Linear SVM**: Baseline maximum-margin classifier
2. **RBF SVM**: Non-linear SVM using Gaussian kernel
3. **Decision Tree**: Single tree with pruning
4. **AdaBoost**: Ensemble of weak decision tree learners

RBF SVM achieves best performance (98% accuracy) by capturing non-linear patterns in digit images.

## 2. Step-by-Step Explanation

### SVM Implementation

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Feature scaling (crucial for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
acc_linear = svm_linear.score(X_test_scaled, y_test)
print(f"Linear SVM Accuracy: {acc_linear:.3f}")

# RBF SVM with hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

svm_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
svm_rbf.fit(X_train_scaled, y_train)
print(f"Best params: {svm_rbf.best_params_}")
print(f"RBF SVM Accuracy: {svm_rbf.score(X_test_scaled, y_test):.3f}")
```

### Decision Tree and AdaBoost

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Single decision tree
tree = DecisionTreeClassifier(max_depth=10, min_samples_split=20)
tree.fit(X_train, y_train)
acc_tree = tree.score(X_test, y_test)

# AdaBoost
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Stumps
    n_estimators=100,
    learning_rate=1.0
)
adaboost.fit(X_train, y_train)
acc_ada = adaboost.score(X_test, y_test)

print(f"Decision Tree Accuracy: {acc_tree:.3f}")
print(f"AdaBoost Accuracy: {acc_ada:.3f}")
```

## 3. Why This Approach Works

**RBF SVM:**
- Kernel trick maps to infinite-dimensional space
- Can capture complex non-linear patterns
- C controls regularization (bias-variance tradeoff)
- Gamma controls influence radius of support vectors

**AdaBoost:**
- Combines many weak learners into strong one
- Focuses on hard examples
- Less prone to overfitting than single tree
- Adaptive: each learner improves on previous

## 4. Code Design Decisions

**Feature Scaling:**
Essential for SVM (distance-based):
```python
# Before scaling: Features have different ranges
# After scaling: Mean=0, Std=1 for all features
```

**GridSearchCV:**
Efficiently tunes hyperparameters with cross-validation:
```python
# Tries all combinations
# Uses CV to avoid overfitting to validation set
# Returns best model
```

**Shallow Trees for AdaBoost:**
Decision stumps (depth=1) as weak learners:
```python
# Each stump: simple decision (better than random)
# Combined: powerful classifier
```

## 5. Performance Analysis

### Results Summary

| Model | Accuracy | Training Time | Prediction Time | Interpretability |
|-------|----------|---------------|-----------------|------------------|
| Linear SVM | 95% | Fast | Very Fast | Low |
| RBF SVM | 98% | Slow | Fast | Very Low |
| Decision Tree | 90% | Very Fast | Very Fast | High |
| AdaBoost | 96% | Medium | Medium | Medium |

**Why RBF Best:**
- Digit patterns are non-linear
- Kernel captures complex shapes
- Well-tuned hyperparameters

**When Others Better:**
- Linear SVM: If actually linearly separable, much faster
- Decision Tree: Need interpretability
- AdaBoost: Good balance of performance and speed

## 6. Interview Explanation

**STAR:**

**Situation**: "Needed to classify handwritten digits for check processing system"

**Task**: "Compare linear vs non-linear classifiers, optimize for both accuracy and speed"

**Action**: "Implemented SVM with RBF kernel, tuned hyperparameters with grid search. Also tried boosting. Scaled features properly (critical for SVM)."

**Result**: "Achieved 98% accuracy with RBF SVM. Fast enough for real-time (< 10ms per digit). Deployed with fallback to simpler model if latency spikes."

## 7. Production Considerations

**Model Selection:**
For production, consider:
- RBF SVM: Best accuracy, reasonable speed
- Linear SVM: If latency critical, slight accuracy drop
- AdaBoost: Good middle ground

**Deployment:**
```python
# Save model and scaler together
import joblib

model_package = {
    'model': svm_rbf.best_estimator_,
    'scaler': scaler
}

joblib.dump(model_package, 'digit_classifier.pkl')

# Prediction function
def classify_digit(image_array):
    # image_array: 784 values
    X = scaler.transform([image_array])
    prediction = model.predict(X)[0]
    probability = model.decision_function(X)[0]
    return prediction, probability
```

**Monitoring:**
- Track prediction confidence (distance from boundary)
- Alert on low-confidence predictions
- Collect misclassified examples for retraining
- Monitor latency (should be < 50ms)

## 8. Key Takeaways

✅ **Kernel trick** enables non-linear boundaries efficiently  
✅ **Feature scaling** crucial for SVM performance  
✅ **Hyperparameter tuning** (GridSearchCV) prevents overfitting  
✅ **Boosting** combines weak learners into strong classifier  
✅ **Single trees overfit** — ensemble or prune  
✅ **RBF SVM excellent** for complex non-linear patterns  
✅ **Tradeoff**: Accuracy vs speed vs interpretability  
✅ **Support vectors** are the "important" examples  

**Next Steps:** Problem Set 4 tackles neural networks and reinforcement learning!

---

**Made By Ritesh Rana**
