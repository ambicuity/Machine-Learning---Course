# Problem Set 4

## 1. Objective

This advanced problem set covers neural networks and reinforcement learning:
- Implement neural networks from scratch
- Understand and code backpropagation
- Build a simple CNN for image classification
- Apply reinforcement learning to a gridworld problem
- Compare deep learning with classical ML methods

Final problem set synthesizes everything you've learned!

## 2. Concepts Covered

- **Neural_Networks/Neural_Networks_Basics**: Architecture, forward pass
- **Neural_Networks/Backpropagation**: Gradient computation via chain rule
- **Neural_Networks/Convolutional_Neural_Networks**: Conv layers, pooling
- **Reinforcement_Learning/RL_Basics**: Agents, environments, rewards
- **Reinforcement_Learning/Value_Iteration**: MDP solving
- All previous concepts (putting it all together)

## 3. Dataset Description

**Part A: MNIST Digit Classification (Full)**
- 60,000 training images, 10,000 test images
- 10 classes (digits 0-9)
- 28x28 grayscale images
- Perfect for neural networks and CNNs

**Part B: GridWorld Navigation**
- 10x10 grid environment
- Agent starts at random position
- Goal: Reach target (reward +10)
- Obstacles: Penalty (-1)
- Actions: Up, Down, Left, Right

## 4. Tasks

### Part A: Neural Networks (6-8 hours)

**Task 1: Implement Feedforward Neural Network**
- 2-layer network: 784 → 128 → 10
- ReLU activation for hidden layer
- Softmax for output layer
- Forward pass implementation

**Task 2: Implement Backpropagation**
- Compute gradients for all layers
- Chain rule application
- Weight updates using gradient descent
- Verify with numerical gradients

**Task 3: Train and Evaluate**
- Train for several epochs
- Track train/validation loss
- Plot learning curves
- Achieve >95% test accuracy

**Task 4: Build Simple CNN**
- Architecture: Conv → Pool → Conv → Pool → FC
- Use PyTorch or TensorFlow
- Compare with feedforward network
- Achieve >98% accuracy

**Task 5: Experiment with Architectures**
- Try different numbers of layers
- Experiment with dropout
- Try different optimizers (SGD, Adam)
- Analyze results

### Part B: Reinforcement Learning (4-6 hours)

**Task 6: Implement Value Iteration**
- Define GridWorld MDP
- Implement Bellman update
- Compute optimal value function
- Extract optimal policy

**Task 7: Analyze Results**
- Visualize value function as heatmap
- Show optimal policy with arrows
- Analyze convergence speed
- Test in different environments

## 5. Expected Output

### Neural Network Performance
- Feedforward NN: 96-97% accuracy
- CNN: 98-99% accuracy  
- Training curves showing convergence
- Confusion matrix analysis

### RL Performance
- Value iteration converges in <100 iterations
- Optimal policy finds shortest path
- Value function properly reflects rewards
- Agent successfully navigates grid

### Analysis
- Why does CNN outperform feedforward?
- How does RL differ from supervised learning?
- When to use each approach?

## 6. Interview Perspective

**Key Questions:**

1. **"Explain backpropagation"**
   - Chain rule to compute gradients
   - Backward pass through network
   - Efficient: Single forward + backward pass

2. **"What's the vanishing gradient problem?"**
   - Gradients become very small in deep networks
   - Sigmoids/tanh saturate → near-zero gradients
   - Solutions: ReLU, residual connections, batch norm

3. **"CNN vs regular neural network?"**
   - CNN: Parameter sharing, local connections
   - Regular: Fully connected, many parameters
   - CNN better for images (translation invariance)

4. **"What's the Bellman equation?"**
   - V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
   - Value of state = immediate reward + discounted future value
   - Foundation of RL

5. **"Supervised learning vs RL?"**
   - Supervised: Learn from labeled data
   - RL: Learn from rewards through trial and error
   - RL: Delayed rewards, exploration-exploitation

## 7. Common Mistakes

**Neural Networks:**
- Not shuffling training data
- Learning rate too high (divergence)
- Not using activation functions
- Forgetting bias terms
- Not validating gradients
- Overfitting (not using regularization)

**RL:**
- Not converging value iteration properly
- Wrong discount factor (γ)
- Not handling terminal states correctly
- Confusing policy and value function
- Not exploring enough (if learning from experience)

## 8. Extension Ideas (Optional)

**Neural Networks:**
- Implement batch normalization
- Try data augmentation
- Implement ResNet architecture
- Transfer learning from pretrained model

**Reinforcement Learning:**
- Implement Q-learning
- Try policy gradient methods
- Add stochastic transitions
- Solve more complex environment (cartpole, mountain car)

## 9. Data Files

From `Data/PS4-data.zip`:
- `mnist_train.csv`: Training digits
- `mnist_test.csv`: Test digits
- `gridworld_config.txt`: Environment specification

## 10. Submission Checklist

- [ ] Neural network implemented from scratch
- [ ] Backpropagation working correctly
- [ ] CNN trained and evaluated
- [ ] Value iteration implemented
- [ ] All visualizations included
- [ ] Thorough analysis of results
- [ ] Understood deep learning fundamentals

**Time estimate**: 10-14 hours (most challenging problem set)

**Final note**: This problem set is the culmination of your ML journey. Take your time, understand deeply, and celebrate when you finish! 🎉

---

**Made By Ritesh Rana**
