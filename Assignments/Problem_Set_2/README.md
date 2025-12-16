# Problem Set 2

## 1. Objective

This problem set focuses on classification algorithms, teaching you to:
- Implement logistic regression for binary classification
- Apply Gaussian Discriminant Analysis (GDA) 
- Build a Naive Bayes classifier
- Compare generative vs discriminative models
- Handle real-world classification problems

You'll work with email spam detection, applying multiple classification approaches and understanding their tradeoffs.

## 2. Concepts Covered

This assignment reinforces:
- **Supervised_Learning/Logistic_Regression**: Binary classification fundamentals
- **Probabilistic_Models/Gaussian_Discriminant_Analysis**: Generative classification
- **Probabilistic_Models/Naive_Bayes**: Probabilistic classification with independence assumption
- **Probabilistic_Models/Laplace_Smoothing**: Handling zero probabilities
- **Model_Evaluation/Evaluation_Metrics**: Precision, recall, F1, ROC-AUC

## 3. Dataset Description

**Dataset: Email Spam Detection**

Classify emails as spam or not spam based on text features.

**Features:**
- Word frequencies: Frequency of common words (e.g., "free", "click", "money")
- Character frequencies: Special characters like $, !, @
- Capital letter sequences: Length of longest continuous capital letters
- ~50 features total (bag-of-words representation)

**Target:**
- `spam`: 1 = spam, 0 = not spam

**Real-world Analogy**: Like Gmail's spam filter, protecting users from unwanted emails.

**Data Characteristics:**
- ~4000 training examples
- Imbalanced (more non-spam than spam)
- Sparse features (many zeros)
- Text-based features (counts and frequencies)

## 4. Tasks

### Task 1: Exploratory Data Analysis
- Load dataset from `Data/PS2-data.zip`
- Check class balance (spam vs not spam)
- Visualize feature distributions for each class
- Identify most discriminative features

### Task 2: Implement Logistic Regression
- Implement sigmoid function
- Implement log-loss cost function
- Train using gradient descent
- Evaluate on test set

### Task 3: Implement Gaussian Discriminant Analysis
- Fit Gaussian distributions for each class
- Estimate means and covariances
- Make predictions using Bayes' theorem
- Compare with logistic regression

### Task 4: Implement Naive Bayes
- Assume feature independence
- Estimate class priors and feature probabilities
- Apply Laplace smoothing
- Make predictions using Bayes' theorem

### Task 5: Model Comparison
- Compare all three models on:
  - Accuracy, Precision, Recall, F1
  - ROC curves and AUC
  - Training time
  - Interpretability
- Analyze when each model excels

### Task 6: Error Analysis
- Which emails are misclassified?
- False positives vs false negatives
- Cost-benefit analysis (which error is worse?)
- Threshold tuning for business needs

## 5. Expected Output

### Performance Metrics
- Logistic Regression: ~95% accuracy, 0.97 AUC
- GDA: ~92% accuracy, 0.95 AUC
- Naive Bayes: ~93% accuracy, 0.96 AUC

### Visualizations
- Class distributions
- ROC curves for all models
- Confusion matrices
- Feature importance

### Analysis
- When does each model work best?
- Tradeoffs between models
- Recommendations for deployment

## 6. Interview Perspective

**Key Questions:**

1. **"What's the difference between generative and discriminative models?"**
   - Generative (GDA, NB): Model P(X|Y) and P(Y), use Bayes' rule
   - Discriminative (Logistic): Model P(Y|X) directly
   - Generative needs fewer examples but makes stronger assumptions

2. **"Why use Naive Bayes despite the 'naive' assumption?"**
   - Works surprisingly well even when independence violated
   - Fast training and prediction
   - Handles high dimensions well
   - Good with small datasets

3. **"How do you handle class imbalance?"**
   - Adjust decision threshold
   - Use class weights
   - Oversample minority or undersample majority
   - Use appropriate metrics (F1, not accuracy)

4. **"Explain logistic regression vs linear regression"**
   - Logistic: Classification, sigmoid, log-loss, outputs probabilities
   - Linear: Regression, identity, MSE, outputs continuous values

## 7. Common Mistakes

- Using accuracy for imbalanced data (misleading)
- Not applying Laplace smoothing in Naive Bayes (zero probabilities)
- Forgetting GDA assumes Gaussian distributions (check this!)
- Not tuning threshold for business costs
- Comparing models only on accuracy (need precision/recall)

## 8. Extension Ideas (Optional)

- Implement Multinomial Naive Bayes for text
- Try feature selection to improve performance
- Implement different variants of GDA (different covariance assumptions)
- Build an ensemble combining all three models
- Deploy as a REST API for real-time spam detection

## 9. Data Files

Extract from `Data/PS2-data.zip`:
- `spam_train.csv`: Training data with features and labels
- `spam_test.csv`: Test data
- `feature_names.txt`: Description of features

## 10. Submission Checklist

- [ ] All three models implemented
- [ ] Performance comparison completed
- [ ] Error analysis documented
- [ ] Visualizations included
- [ ] Code clean and commented
- [ ] Understood tradeoffs between models

**Time estimate**: 5-7 hours

---

**Made By Ritesh Rana**
