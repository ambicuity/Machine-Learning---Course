# Generalized Linear Models (GLMs)

## 1. What is this concept?

Generalized Linear Models (GLMs) are a flexible framework that extends linear regression to handle different types of output variables. While linear regression assumes outputs are continuous and normally distributed, GLMs can model binary outcomes, counts, positive values, and more.

**Simple Analogy**: Think of GLMs as a toolkit where linear regression is just one tool. Need to predict yes/no? Use logistic regression (a GLM). Need to predict counts (number of claims)? Use Poisson regression (a GLM). Same framework, different "link functions."

## 2. Why do we need it?

Linear regression has limitations:
- Assumes output is continuous and unbounded
- Assumes constant variance
- Can predict impossible values (negative counts, probabilities > 1)

**Problems GLMs Solve:**
- **Binary outcomes**: Classification (logistic regression)
- **Count data**: Number of events (Poisson regression)
- **Positive continuous**: Prices, durations (Gamma regression)
- **Proportions**: Success rates (Beta regression)
- **Unify framework**: One theory covers many models

**Real Example**: Predicting number of insurance claims (must be non-negative integer) — linear regression could predict -2.5 claims, but Poisson regression naturally handles counts!

## 3. Mathematical Intuition (No heavy math)

**Three Components of a GLM:**

**1. Random Component (Distribution Family):**
- Specifies distribution of Y given X
- Examples: Gaussian (normal), Bernoulli, Poisson, Gamma
- From exponential family of distributions

**2. Systematic Component (Linear Predictor):**
```
η = θᵀx = θ₀ + θ₁x₁ + ... + θₙxₙ
```
- Same as linear regression: weighted combination of features

**3. Link Function:**
```
g(E[Y|X]) = η
```
- Connects the expected value of Y to the linear predictor
- Different problems need different links

**Common GLMs:**

**Linear Regression:**
- Distribution: Gaussian
- Link: Identity (g(μ) = μ)
- Model: E[Y|X] = θᵀx

**Logistic Regression:**
- Distribution: Bernoulli
- Link: Logit (g(μ) = log(μ/(1-μ)))
- Model: P(Y=1|X) = 1/(1 + e^(-θᵀx))

**Poisson Regression:**
- Distribution: Poisson
- Link: Log (g(μ) = log(μ))
- Model: E[Y|X] = e^(θᵀx)

**Why Link Functions Matter:**
- Keep predictions in valid range (probabilities in [0,1], counts ≥ 0)
- Transform the problem to linear space
- Allow using same optimization techniques

## 4. How it works step-by-step

**General GLM Algorithm:**

1. **Choose Distribution**: Based on output type
   - Binary → Bernoulli
   - Counts → Poisson
   - Continuous → Gaussian

2. **Choose Link Function**: 
   - Usually the "canonical" link for that distribution
   - Logit for Bernoulli, Log for Poisson, Identity for Gaussian

3. **Specify Model**:
   ```
   g(E[Y|X]) = θᵀx
   ```

4. **Maximum Likelihood Estimation**:
   - Write likelihood function based on chosen distribution
   - Maximize likelihood to find best θ
   - Usually use iterative methods (Newton-Raphson, IRLS)

5. **Predict**:
   - Calculate linear predictor: η = θᵀx
   - Apply inverse link: E[Y|X] = g⁻¹(η)

**Example: Poisson Regression**

1. Distribution: P(Y=k) = (λᵏe^(-λ))/k! where λ = E[Y|X]
2. Link: log(λ) = θᵀx
3. Inverse link: λ = e^(θᵀx)
4. Predict: Expected count = e^(θᵀx)

## 5. Real-world use cases

**Industry Applications:**

- **Insurance**:
  - Poisson regression: Number of claims per customer
  - Gamma regression: Claim amounts
  - Logistic regression: Policy cancellation

- **Healthcare**:
  - Logistic: Disease diagnosis (diseased/healthy)
  - Poisson: Hospital admissions count
  - Gamma: Length of stay (positive, right-skewed)

- **Marketing**:
  - Poisson: Number of purchases in time period
  - Logistic: Customer conversion (buy/don't buy)
  - Gamma: Customer lifetime value

- **Finance**:
  - Logistic: Credit default (default/no default)
  - Poisson: Number of transactions
  - Gamma: Loss amount given default

- **Environmental Science**:
  - Poisson: Species count at locations
  - Gamma: Rainfall amount (positive only)

- **Web Analytics**:
  - Poisson: Click count per session
  - Logistic: Conversion (click/no click)

**Why Companies Use GLMs:**
- Predictions always in valid range
- Interpretable coefficients
- Handles different data types naturally
- Well-understood statistical properties
- Fast and stable training

## 6. How to implement in real life

**Data Requirements:**
- Choose distribution based on output type
- Sufficient data for MLE to work well
- Check for overdispersion (variance > mean for Poisson)
- Handle zero-inflation if needed

**Tools & Libraries:**

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

# Logistic Regression (GLM with Bernoulli family)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Poisson Regression (for count data)
poisson_model = PoissonRegressor()
poisson_model.fit(X_train, y_train)
predictions = poisson_model.predict(X_test)  # Predicted counts

# Using statsmodels for more GLM options
import statsmodels.api as sm

# Poisson GLM
X_with_intercept = sm.add_constant(X_train)
poisson_glm = sm.GLM(y_train, X_with_intercept, 
                     family=sm.families.Poisson())
result = poisson_glm.fit()
print(result.summary())

# Gamma GLM (for positive continuous data)
gamma_glm = sm.GLM(y_train, X_with_intercept,
                   family=sm.families.Gamma())
result = gamma_glm.fit()

# Negative Binomial (for overdispersed count data)
from statsmodels.discrete.discrete_model import NegativeBinomial
nb_model = NegativeBinomial(y_train, X_with_intercept)
nb_result = nb_model.fit()
```

**Deployment Considerations:**
- Validate distribution assumptions (check residuals)
- Monitor for overdispersion (Poisson may be inadequate)
- Check for zero-inflation in count models
- Ensure predictions stay in valid range
- Consider quasi-likelihood if variance assumption wrong
- Retrain when data distribution changes

## 7. Interview perspective

**Common Interview Questions:**

1. **"What are Generalized Linear Models?"**
   - Framework extending linear models to different output types
   - Three components: distribution, linear predictor, link function
   - Examples: Logistic regression, Poisson regression

2. **"How is logistic regression a GLM?"**
   - Bernoulli distribution family
   - Logit link function
   - Fits the GLM framework perfectly

3. **"When would you use Poisson vs Negative Binomial regression?"**
   - Poisson: Count data with variance ≈ mean
   - Negative Binomial: Count data with overdispersion (variance > mean)
   - Check data: if variance >> mean, use Negative Binomial

4. **"What's a link function and why do we need it?"**
   - Maps expected value to linear predictor
   - Keeps predictions in valid range
   - Example: Log link ensures counts are positive

5. **"Explain exponential family of distributions"**
   - Family of distributions with specific mathematical form
   - Includes: Gaussian, Bernoulli, Poisson, Gamma, etc.
   - GLMs work with exponential family distributions

**How to Explain:**
- Start with linear regression limitations
- Introduce different output types
- Explain link function concept simply
- Give concrete example (count data, binary data)

**Traps to Avoid:**
- Not knowing logistic regression is a GLM
- Confusing link function with activation function
- Using wrong distribution for data type
- Not checking for overdispersion
- Claiming GLMs can handle any distribution (must be exponential family)

## 8. Common mistakes students make

**Conceptual Mistakes:**
- Thinking GLMs are completely different from linear/logistic regression
- Not understanding that logistic regression is just one GLM
- Confusing link function with transformation of Y
- Using linear regression for count/binary data
- Not checking distribution assumptions

**Implementation Mistakes:**
- Using Poisson when data is overdispersed (variance >> mean)
- Ignoring zero-inflation in count data
- Not using appropriate link function for distribution
- Forgetting to add intercept term in statsmodels
- Using wrong evaluation metrics for different GLMs

**Model Selection Mistakes:**
```python
# Bad: Linear regression for count data
model = LinearRegression()
model.fit(X, y_counts)  # Can predict negative counts!

# Good: Poisson regression for count data
model = PoissonRegressor()
model.fit(X, y_counts)  # Always predicts non-negative

# Bad: Poisson for overdispersed data
# Check: variance >> mean

# Good: Negative Binomial for overdispersed data
from statsmodels.discrete.discrete_model import NegativeBinomial
```

## 9. When NOT to use this approach

**Don't use GLMs when:**

- **Non-exponential family distributions**: Some distributions don't fit GLM framework
- **Complex non-linear relationships**: Need neural networks, trees, or kernel methods
- **High-dimensional data**: Regularization needed (Elastic Net GLMs exist)
- **Very flexible modeling needed**: Tree ensembles, deep learning
- **Distribution unknown**: Non-parametric methods better
- **Perfect separation** (logistic): Model may not converge

**Better Alternatives:**
- Complex patterns → Decision trees, random forests, neural networks
- High dimensions → Regularized GLMs, Lasso/Ridge
- Flexible non-parametric → GAMs (Generalized Additive Models)
- Time series → ARIMA, state space models
- Perfect separation → Penalized logistic regression
- Unknown distribution → Non-parametric methods

## 10. Key takeaways

✅ **GLMs extend linear models** to different output types  
✅ **Three components**: Distribution family, linear predictor, link function  
✅ **Logistic regression is a GLM** (Bernoulli + logit link)  
✅ **Poisson regression** for count data  
✅ **Gamma regression** for positive continuous data  
✅ **Link functions** keep predictions in valid range  
✅ **All use MLE** (Maximum Likelihood Estimation)  
✅ **Check assumptions**: Distribution, overdispersion, zero-inflation  
✅ **Exponential family** — family of distributions GLMs work with  
✅ **Interview tip**: Know logistic regression is a special case of GLM  

**Common GLM Types:**
- **Linear Regression**: Gaussian + Identity link
- **Logistic Regression**: Bernoulli + Logit link
- **Poisson Regression**: Poisson + Log link
- **Gamma Regression**: Gamma + Log link
- **Negative Binomial**: Negative Binomial + Log link

---

**Made By Ritesh Rana**
