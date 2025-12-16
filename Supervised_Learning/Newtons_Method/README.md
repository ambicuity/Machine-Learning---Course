# Newton's Method

## 1. What is this concept?

Newton's Method (also called Newton-Raphson Method) is an optimization algorithm that finds parameters faster than gradient descent by using second-order information (curvature) about the cost function. While gradient descent uses only the slope, Newton's Method also considers how the slope is changing.

**Simple Analogy**: Gradient descent is like walking downhill taking small steps based on how steep the hill is. Newton's Method is like predicting where the valley bottom is based on both steepness AND curvature, then jumping there directly.

## 2. Why do we need it?

Gradient descent can be slow, especially when:
- Features have different scales
- Cost function has narrow valleys
- Learning rate is hard to tune

**Problems Newton's Method Solves:**
- Faster convergence (fewer iterations)
- No learning rate to tune
- Better handles poorly scaled features
- Converges in one step for quadratic functions

**Real-World Impact**: Training time reduction from hours to minutes for some problems. Critical for large-scale ML where each iteration is expensive.

## 3. Mathematical Intuition (No heavy math)

**Gradient Descent Update:**
```
θ := θ - α∇J(θ)
```
- Uses first derivative (gradient)
- Step size α needs tuning

**Newton's Method Update:**
```
θ := θ - H⁻¹∇J(θ)
```

Where:
- **∇J(θ)** = gradient (first derivative)
- **H** = Hessian matrix (second derivatives)
- **H⁻¹** = inverse Hessian (curvature information)

**Intuition:**
- Gradient tells us direction to move
- Hessian tells us how quickly gradient changes
- Combining both gives us optimal step size automatically!

**Hessian Matrix:**
```
H[i,j] = ∂²J / ∂θᵢ∂θⱼ
```

- Captures curvature in all directions
- Diagonal elements: curvature along each parameter
- Off-diagonal: interaction between parameters

**For Logistic Regression:**
```
∇J(θ) = (1/m)Xᵀ(h(X) - y)
H = (1/m)XᵀSX
```
Where S is diagonal matrix with s[i,i] = h(x⁽ⁱ⁾)(1 - h(x⁽ⁱ⁾))

## 4. How it works step-by-step

**Newton's Method Algorithm:**

1. **Initialize**: Start with random θ (or zeros)

2. **Calculate Gradient**: 
   ```
   g = ∇J(θ) = (1/m)Xᵀ(h(X) - y)
   ```

3. **Calculate Hessian**:
   ```
   H = (1/m)XᵀSX
   ```
   where S[i,i] = h(x⁽ⁱ⁾)(1 - h(x⁽ⁱ⁾))

4. **Compute Update Direction**:
   ```
   Δθ = -H⁻¹g
   ```
   (Solve Hx = g for x, don't actually compute inverse)

5. **Update Parameters**:
   ```
   θ := θ + Δθ
   ```

6. **Check Convergence**: If ||Δθ|| < ε, stop; else repeat from step 2

**Key Differences from Gradient Descent:**
- No learning rate needed
- Uses second derivatives
- Fewer iterations but each iteration costlier
- Can take huge steps when curvature is small

## 5. Real-world use cases

**Industry Applications:**

- **Machine Learning Libraries**: 
  - Scikit-learn uses Newton's Method for Logistic Regression by default
  - Statsmodels uses it for generalized linear models

- **Financial Modeling**: 
  - Option pricing (Black-Scholes)
  - Portfolio optimization
  - Risk assessment models

- **Statistics**: 
  - Maximum likelihood estimation
  - Generalized linear models
  - Survival analysis

- **Neural Networks**: 
  - L-BFGS (quasi-Newton method) for small networks
  - Second-order optimization research

- **Robotics**: 
  - Inverse kinematics
  - Control optimization

- **Scientific Computing**: 
  - Root finding
  - Equation solving

**Why Companies Use It:**
- Faster training for moderate-sized problems
- No hyperparameter tuning (no learning rate)
- Default in many statistical packages
- Well-understood convergence properties

## 6. How to implement in real life

**Data Requirements:**
- Same as gradient descent methods
- Works best with moderate number of features (n < 10,000)
- For large n, computing and inverting Hessian becomes prohibitive

**Tools & Libraries:**

```python
# Scikit-learn uses Newton's Method by default for Logistic Regression
from sklearn.linear_model import LogisticRegression

# solver='newton-cg' explicitly uses Newton's Conjugate Gradient
model = LogisticRegression(solver='newton-cg')
model.fit(X_train, y_train)

# For custom implementation
import numpy as np

def newton_method_logistic(X, y, max_iter=10):
    m, n = X.shape
    theta = np.zeros(n)
    
    for i in range(max_iter):
        # Forward pass
        z = X @ theta
        h = 1 / (1 + np.exp(-z))
        
        # Gradient
        gradient = (1/m) * X.T @ (h - y)
        
        # Hessian
        S = np.diag(h * (1 - h))
        H = (1/m) * X.T @ S @ X
        
        # Update (solve linear system instead of inverting)
        delta = np.linalg.solve(H, gradient)
        theta = theta - delta
        
        # Check convergence
        if np.linalg.norm(delta) < 1e-6:
            break
    
    return theta

# Using scipy for optimization
from scipy.optimize import minimize

def cost_function(theta, X, y):
    h = 1 / (1 + np.exp(-X @ theta))
    return -np.mean(y * np.log(h) + (1-y) * np.log(1-h))

result = minimize(cost_function, x0=np.zeros(n), args=(X, y), 
                  method='Newton-CG', jac=gradient_function)
```

**Deployment Considerations:**
- Faster training time (fewer iterations)
- Higher memory usage (store Hessian matrix)
- Each iteration more expensive computationally
- Not suitable for very high dimensions
- Works well for convex problems (logistic regression, linear models)
- Can diverge if initialization is poor

## 7. Interview perspective

**Common Interview Questions:**

1. **"What's the difference between Gradient Descent and Newton's Method?"**
   - GD: First-order, needs learning rate, more iterations, cheaper per iteration
   - Newton: Second-order, no learning rate, fewer iterations, expensive per iteration

2. **"Why is Newton's Method faster?"**
   - Uses curvature information (Hessian)
   - Automatically determines optimal step size
   - Converges quadratically near optimum

3. **"What's the computational complexity?"**
   - GD: O(nd) per iteration (n examples, d features)
   - Newton: O(nd²) for Hessian, O(d³) for inversion
   - Newton impractical when d is very large

4. **"When would you use Newton's Method over Gradient Descent?"**
   - Moderate number of features (< 10,000)
   - Need fast convergence
   - Don't want to tune learning rate
   - Problem is convex (logistic regression)

5. **"What are quasi-Newton methods?"**
   - Approximate Hessian (don't compute exactly)
   - BFGS, L-BFGS are popular variants
   - Better scaling for larger problems

**How to Explain:**
- Start with gradient descent limitations
- Introduce curvature concept simply
- Mention tradeoff: fewer iterations but more expensive
- Give complexity analysis

**Traps to Avoid:**
- Saying it's always better (not for high dimensions)
- Not mentioning memory/computational costs
- Forgetting that Hessian can be singular
- Not knowing about quasi-Newton approximations

## 8. Common mistakes students make

**Conceptual Mistakes:**
- Thinking Newton's Method is always faster (depends on problem size)
- Not understanding it's for optimization, not just root finding
- Confusing iterations with total computation time
- Believing it works for non-convex problems (can get stuck)
- Not recognizing it's what scikit-learn uses by default

**Implementation Mistakes:**
- Actually computing matrix inverse (numerically unstable)
  - Should solve: Hx = g instead of x = H⁻¹g
- Not checking if Hessian is positive definite
- Using for very high-dimensional problems (millions of features)
- Not handling singular or near-singular Hessian
- Forgetting to vectorize operations (slow implementations)

**Practical Mistakes:**
- Using for neural networks with millions of parameters
- Not considering memory constraints
- Ignoring that each iteration is much more expensive
- Not using quasi-Newton methods when appropriate

## 9. When NOT to use this approach

**Don't use Newton's Method when:**

- **High-dimensional problems**: Computing/storing d×d Hessian prohibitive (deep learning)
- **Non-convex optimization**: Can converge to saddle points or diverge
- **Hessian is singular**: Inverse doesn't exist
- **Memory constraints**: Hessian matrix too large to store
- **Online learning**: Need to update with streaming data
- **Very large datasets**: Computing Hessian over all data too expensive
- **Stochastic optimization**: Mini-batch training scenarios

**Better Alternatives:**
- High dimensions → L-BFGS, Adam, SGD
- Deep learning → Adam, RMSprop, SGD with momentum
- Online learning → Stochastic gradient descent variants
- Large-scale → Mini-batch SGD, distributed optimization
- Non-convex → Gradient descent with careful initialization
- Limited memory → First-order methods

**When Newton's Method Shines:**
- Convex problems with moderate dimensions
- Logistic regression with < 10,000 features
- Statistical models (GLMs)
- When you need fast convergence
- When you don't want to tune learning rate

## 10. Key takeaways

✅ **Newton's Method uses second derivatives** (Hessian) for faster convergence  
✅ **No learning rate needed** — step size automatic from curvature  
✅ **Fewer iterations** than gradient descent (quadratic convergence)  
✅ **Each iteration expensive**: O(d³) for Hessian inversion  
✅ **Best for moderate dimensions** (d < 10,000)  
✅ **Default in scikit-learn** for logistic regression  
✅ **Memory intensive**: Stores d×d matrix  
✅ **Quasi-Newton methods** (L-BFGS) approximate Hessian for larger problems  
✅ **Works best on convex problems** like logistic regression  
✅ **Interview tip**: Compare with GD on iterations vs cost per iteration  

**Next Section**: Move to Math Refresher to strengthen foundations in Linear Algebra and Probability!

---

**Made By Ritesh Rana**
