# Weighted Least Squares

## 1. What is this concept?

Weighted Least Squares (WLS) is a variant of linear regression where different data points have different levels of importance or reliability. Instead of treating all errors equally, we give more weight to some predictions and less to others.

**Simple Analogy**: Imagine learning to predict temperature using both highly accurate weather stations and cheap home thermometers. You'd trust the professional stations more. WLS lets you do exactly that — give more importance to reliable data points.

## 2. Why do we need it?

Standard linear regression assumes all data points are equally reliable and important. But real-world data often violates this:

**Problems WLS Solves:**
- **Heteroscedasticity**: When error variance isn't constant (some predictions are naturally noisier)
- **Different reliability**: Some measurements are more accurate than others
- **Varying importance**: Some predictions matter more for your application
- **Local learning**: Emphasize nearby points when making predictions (locally weighted regression)

**Real Example**: In financial data, recent transactions might be more relevant than old ones. In sensor data, some sensors are more accurate than others.

## 3. Mathematical Intuition (No heavy math)

**Standard Cost Function:**
```
J(θ) = (1/2) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

**Weighted Cost Function:**
```
J(θ) = (1/2) Σ w⁽ⁱ⁾(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

Key difference: **w⁽ⁱ⁾** is the weight for example i

- **w⁽ⁱ⁾ = 2**: This error counts twice as much
- **w⁽ⁱ⁾ = 0.5**: This error counts half as much
- **w⁽ⁱ⁾ = 0**: Ignore this point completely

**Locally Weighted Regression (LOESS/LOWESS):**

For predicting at query point x:
```
w⁽ⁱ⁾ = exp(-(x⁽ⁱ⁾ - x)²/(2τ²))
```

- Points close to x get high weight
- Points far from x get low weight
- **τ** (tau) controls how quickly weight decreases with distance

## 4. How it works step-by-step

**Standard Weighted Least Squares:**

1. **Assign Weights**: Determine w⁽ⁱ⁾ for each training example
   - Based on measurement reliability
   - Based on data quality
   - Based on business importance
   
2. **Weighted Cost**: Modify cost function to include weights

3. **Optimize**: Use weighted gradient descent or closed-form solution
   ```
   θ = (XᵀWX)⁻¹XᵀWy
   ```
   where W is diagonal matrix of weights

4. **Predict**: Use learned θ just like regular linear regression

**Locally Weighted Regression (LWR):**

1. **For each prediction**: 
   - Calculate weights based on distance to query point
   - Nearby points get high weight, far points get low weight

2. **Fit Local Model**: Train a weighted linear regression using these weights

3. **Make Prediction**: Use this local model for just this prediction

4. **Repeat**: For next prediction, recalculate weights and refit

**Key Difference**: LWR fits a new model for EACH prediction!

## 5. Real-world use cases

**Industry Applications:**

- **Finance**: 
  - Time-series prediction (recent data more relevant)
  - Risk modeling (weight by transaction size)
  
- **Healthcare**: 
  - Weight by measurement accuracy (lab vs home tests)
  - Patient similarity for personalized predictions

- **Marketing**: 
  - Customer lifetime value (weight by recency)
  - Sales forecasting (weight by data quality)

- **Manufacturing**: 
  - Quality control (weight by inspector experience)
  - Sensor fusion (weight by sensor accuracy)

- **Climate Science**: 
  - Temperature prediction (weight by station calibration)
  - Pollution modeling (weight by sensor type)

- **Recommendation Systems**: 
  - Weight by user engagement, purchase value
  - Emphasize recent behavior over old patterns

**Locally Weighted Regression Use Cases:**
- Non-parametric predictions when relationships aren't globally linear
- Robot control and navigation
- Financial trading strategies
- Any situation where local patterns differ from global ones

## 6. How to implement in real life

**Data Requirements:**
- Need to determine appropriate weights (domain knowledge crucial)
- For LWR: Enough data for each local region
- Computational resources for LWR (memory-based, slow at prediction)

**Tools & Libraries:**

```python
# Standard Weighted Least Squares
from sklearn.linear_model import LinearRegression

# Create weights (example: inverse of variance)
weights = 1.0 / np.var(residuals, axis=0)

# Fit with sample_weight parameter
model = LinearRegression()
model.fit(X_train, y_train, sample_weight=weights)

# Locally Weighted Regression (custom implementation)
def lowess_predict(X_train, y_train, x_query, tau):
    # Calculate weights based on distance
    weights = np.exp(-np.sum((X_train - x_query)**2, axis=1) / (2 * tau**2))
    
    # Fit weighted model
    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=weights)
    
    # Predict
    return model.predict(x_query.reshape(1, -1))

# Use statsmodels for LOWESS
from statsmodels.nonparametric.smoothers_lowess import lowess
smoothed = lowess(y, X, frac=0.3)  # frac controls bandwidth
```

**Deployment Considerations:**

**For Standard WLS:**
- Store weights with training data
- Fast prediction (just like linear regression)
- Easy to update with new data

**For LWR:**
- Must store entire training set (memory-based)
- Slow predictions (refit for each query)
- Bandwidth parameter (τ) needs tuning
- Not suitable for real-time high-volume predictions
- Consider approximate methods for large-scale deployment

## 7. Interview perspective

**Common Interview Questions:**

1. **"What's the difference between WLS and regular linear regression?"**
   - Good: "WLS assigns different importance to training examples, useful when data reliability varies or we want local predictions"
   - Show you understand both when to use it and the computational tradeoffs

2. **"Explain Locally Weighted Regression"**
   - It's non-parametric (no fixed parameters)
   - Fits a new model for each prediction
   - Uses nearby points more heavily
   - Tradeoff: Flexible but computationally expensive

3. **"How do you choose weights?"**
   - Domain knowledge (measurement accuracy, data quality)
   - Inverse variance (higher variance → lower weight)
   - Recency (time-decaying weights)
   - For LWR: Based on distance with bandwidth parameter τ

4. **"What's the computational complexity?"**
   - Standard WLS: Same as linear regression (fast prediction)
   - LWR: O(n) per prediction (stores all training data, slow)

**How to Explain:**
- Start with why equal weighting might be wrong
- Give concrete example (sensor accuracy, data quality)
- Mention tradeoff between flexibility and computation
- Discuss when you'd use WLS vs LWR

**Traps to Avoid:**
- Confusing WLS with regularization (different purposes)
- Not mentioning LWR is non-parametric
- Forgetting computational cost of LWR at prediction time
- Not discussing how to choose weights in practice

## 8. Common mistakes students make

**Conceptual Mistakes:**
- Thinking weights are learnable parameters (they're predetermined)
- Confusing weights with feature importance
- Using LWR when linear regression would suffice
- Not understanding difference between parametric (WLS) and non-parametric (LWR)

**Implementation Mistakes:**
- Setting weights arbitrarily without justification
- Making all weights sum to 1 when that's not necessary
- For LWR: Choosing τ (bandwidth) too small (overfitting) or too large (underfitting)
- Not normalizing features before calculating distances in LWR
- Forgetting LWR needs to store all training data

**Performance Mistakes:**
- Using LWR for large-scale production systems without considering latency
- Not caching or approximating LWR predictions
- Treating LWR as a parametric model

## 9. When NOT to use this approach

**Standard WLS:**
- When all data points are equally reliable
- When you can't justify weight assignments
- When weights would be arbitrary or subjective
- Better to collect better quality data than compensate with weights

**Locally Weighted Regression:**
- **High-dimensional data**: Distance becomes meaningless (curse of dimensionality)
- **Real-time predictions**: Too slow for low-latency requirements
- **Large datasets**: Memory requirements prohibitive
- **Global linear relationships**: Regular regression is simpler and sufficient
- **Interpretability needed**: Hard to explain "trained many local models"
- **Limited training data**: Not enough points for each local region

**Better Alternatives:**
- LWR too slow → Use parametric models (polynomial regression, neural networks)
- High dimensions → Use regularized linear models or feature selection
- Non-linear globally → Use kernel methods, decision trees, or neural networks
- Need speed → Approximate methods or parametric alternatives

## 10. Key takeaways

✅ **WLS treats training examples differently** based on importance or reliability  
✅ **Standard WLS**: Predetermined weights, fast prediction, parametric model  
✅ **LWR**: Distance-based weights, slow prediction, non-parametric model  
✅ **Use WLS when**: Data reliability varies, heteroscedasticity, importance differs  
✅ **Use LWR when**: Local patterns differ from global, non-linear relationships  
✅ **Bandwidth τ**: Controls locality (small = more local, large = more global)  
✅ **Computational tradeoff**: LWR is flexible but expensive  
✅ **Memory requirement**: LWR must store all training data  
✅ **Interview tip**: Discuss both variants and their tradeoffs  
✅ **Production consideration**: LWR often impractical for large-scale deployment  

**Next Topic**: Move to Logistic Regression to learn how to adapt linear models for classification problems!

---

**Made By Ritesh Rana**
