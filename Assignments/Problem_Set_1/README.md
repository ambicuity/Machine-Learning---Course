# Problem Set 1

## 1. Objective

This problem set introduces you to the fundamentals of supervised learning through linear regression. You will learn to:
- Implement linear regression from scratch using gradient descent
- Work with real-world datasets requiring preprocessing
- Understand the mathematics behind least mean squares (LMS)
- Evaluate model performance using appropriate metrics
- Debug and improve model predictions

By the end, you'll have hands-on experience with the entire ML pipeline: data loading, preprocessing, training, evaluation, and interpretation.

## 2. Concepts Covered

This assignment reinforces concepts from:
- **Supervised_Learning/Linear_Regression_LMS**: Core algorithm and implementation
- **Math_Refresher/Linear_Algebra**: Vector operations and matrix multiplication
- **Math_Refresher/Probability**: Understanding distributions and noise
- **Model_Evaluation/Dataset_Splitting**: Train/test splits
- **Model_Evaluation/Evaluation_Metrics**: MSE, RMSE, R² score

## 3. Dataset Description

**Dataset: Housing Prices Prediction**

The dataset contains information about houses and their sale prices. Your goal is to predict house prices based on various features.

**Features:**
- `square_feet`: Total living area in square feet
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `lot_size`: Size of the lot in square feet
- `year_built`: Year the house was constructed
- `garage_spaces`: Number of garage spaces

**Target:**
- `price`: Sale price of the house in dollars

**Real-world Analogy**: Like Zillow's Zestimate, you're building a model that estimates home values based on property characteristics.

**Data Characteristics:**
- ~1000 training examples
- 6 input features
- Continuous target variable
- Some missing values (require imputation)
- Features on different scales (require normalization)

## 4. Tasks

Work through these tasks systematically:

### Task 1: Data Exploration (30 minutes)
- Load the dataset from `Data/PS1-data.zip`
- Check for missing values and outliers
- Visualize distributions of features and target
- Compute correlations between features and price
- **Think**: Which features seem most predictive?

### Task 2: Data Preprocessing (30 minutes)
- Handle missing values (imputation or removal)
- Normalize/standardize features (important for gradient descent)
- Split data into training (80%) and test (20%) sets
- **Think**: Why is feature scaling necessary?

### Task 3: Implement Linear Regression (1-2 hours)
- Implement gradient descent algorithm
- Initialize parameters (θ) to zeros or small random values
- Compute predictions: h(x) = θᵀx
- Calculate cost function: J(θ) = (1/2m)Σ(h(x) - y)²
- Update parameters: θ := θ - α∇J(θ)
- **Think**: How do you choose learning rate α?

### Task 4: Train and Evaluate (30 minutes)
- Train model on training set
- Plot cost vs iterations (should decrease)
- Evaluate on test set using MSE, RMSE, R²
- Compare predictions vs actual prices (scatter plot)
- **Think**: Is the model overfitting or underfitting?

### Task 5: Feature Engineering (Optional - 30 minutes)
- Try adding polynomial features (e.g., square_feet²)
- Try feature interactions (e.g., bedrooms × bathrooms)
- Compare performance with original features
- **Think**: Does complexity always help?

### Task 6: Analysis and Interpretation (30 minutes)
- Which features have highest weights (most important)?
- What is the physical interpretation of each weight?
- Where does the model fail (large errors)?
- How could you improve the model?

## 5. Expected Output

Your submission should include:

### Code
- Clean, well-commented Python code
- Functions for: data loading, preprocessing, training, evaluation
- Visualization code for plots

### Results
- **Training curve**: Cost vs iterations
- **Performance metrics**: MSE, RMSE, R² on train and test sets
- **Prediction plot**: Actual vs predicted prices
- **Feature importance**: Bar chart of learned weights

### Insights
- Which features are most important?
- Does the model generalize well (train vs test performance)?
- Where does it make the largest errors?
- Suggestions for improvement

### Typical Performance
- R² score on test set: 0.75-0.85 (with proper implementation)
- RMSE: $50,000-$80,000 (depends on price range in dataset)

## 6. Interview Perspective

Interviewers often ask about linear regression implementations. Be ready to discuss:

### Technical Questions
1. **"Explain how gradient descent works"**
   - Iteratively update parameters in direction that reduces cost
   - Learning rate controls step size
   - Converges to local (global for convex) minimum

2. **"Why do we need feature scaling?"**
   - Features on different scales lead to elongated cost function
   - Gradient descent converges much slower without scaling
   - Some features dominate if not scaled

3. **"How do you choose learning rate?"**
   - Too large: Divergence (cost increases)
   - Too small: Slow convergence
   - Try multiple values: 0.001, 0.01, 0.1, 1.0
   - Plot cost vs iterations to diagnose

4. **"What's the difference between batch and stochastic gradient descent?"**
   - Batch: Use all data for each update (stable, slow)
   - Stochastic: Use one example per update (fast, noisy)
   - Mini-batch: Middle ground (common in practice)

### How to Walk Through Your Solution
1. **Problem understanding**: "Predict house prices from features"
2. **Data preprocessing**: "Handled missing values, normalized features"
3. **Model choice**: "Linear regression appropriate for continuous target"
4. **Implementation**: "Gradient descent with vectorized operations"
5. **Evaluation**: "R² of 0.8 on test set shows good generalization"
6. **Insights**: "Square footage most important feature, price roughly $200/sqft"

### Common Follow-ups
- "How would you handle outliers?"
- "What if you had millions of data points?"
- "How would you know if you need more data?"
- "Can you derive the gradient mathematically?"

## 7. Common Mistakes

### Data Mistakes
- **Not normalizing features**: Model converges very slowly or diverges
- **Data leakage**: Normalizing before splitting (use train statistics only)
- **Wrong missing value handling**: Using test set info in imputation

### Implementation Mistakes
- **Using loops instead of vectorization**: 100x slower
- **Wrong gradient computation**: Check mathematically and with numerical gradient
- **Not adding intercept term**: Forgetting θ₀ or x₀=1
- **Learning rate issues**: Too large (divergence) or too small (slow)

### Conceptual Mistakes
- **Training on all data**: Need holdout test set
- **Looking at training error only**: Overfitting not detected
- **Random weight initialization**: Not necessary for linear regression, zeros work
- **Expecting perfect predictions**: Real data has noise

### Debugging Tips
```python
# If cost is increasing: Learning rate too high
if cost[i] > cost[i-1]:
    print("Reduce learning rate!")

# If cost not decreasing: Check gradient computation
# Verify with numerical gradient:
numerical_grad = (cost(theta + epsilon) - cost(theta - epsilon)) / (2*epsilon)

# If predictions are way off: Check feature scaling
print(f"Feature means: {X.mean(axis=0)}")
print(f"Feature stds: {X.std(axis=0)}")
```

## 8. Extension Ideas (Optional)

Want to go further? Try these:

### Algorithmic Extensions
- Implement closed-form solution: θ = (XᵀX)⁻¹Xᵀy
- Compare with gradient descent (speed and result)
- Implement stochastic gradient descent (SGD)
- Add regularization (Ridge regression)

### Analysis Extensions
- Learning curves: Plot error vs training set size
- Feature importance: Permutation importance
- Residual analysis: Plot residuals vs predictions
- Cross-validation: K-fold instead of single split

### Real-world Considerations
- How would you deploy this model?
- How would you handle new data with different distribution?
- How would you explain predictions to non-technical stakeholders?
- How would you monitor model performance over time?

## 9. Getting Started

```python
# Suggested imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
import zipfile
with zipfile.ZipFile('Data/PS1-data.zip', 'r') as zip_ref:
    zip_ref.extractall('Data/')

data = pd.read_csv('Data/housing_prices.csv')

# Your implementation here...
```

## 10. Submission Checklist

Before submitting or moving to the solution:

- [ ] Code runs without errors
- [ ] All tasks completed
- [ ] Visualizations included
- [ ] Results documented
- [ ] Code is clean and commented
- [ ] Attempted all required tasks
- [ ] Understood the approach (not just copied code)

**Time estimate**: 4-6 hours for complete implementation

Good luck! Remember: The goal is learning, not perfection. Make mistakes, debug them, and understand why solutions work.

---

**Made By Ritesh Rana**
