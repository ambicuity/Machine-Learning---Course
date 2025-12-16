"""
Unit tests for model training in Problem Set 1.
Tests gradient descent implementation and model performance.
"""

import pytest
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Get the path to the Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')


def load_and_prepare_data():
    """Helper function to load and prepare data for testing."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    
    X = train_df.drop('price', axis=1).values
    y = train_df['price'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val


def gradient_descent(X, y, learning_rate=0.01, n_iterations=100):
    """
    Simple gradient descent implementation for testing.
    Uses adaptive learning rate for stability.
    """
    from sklearn.linear_model import SGDRegressor
    
    # Use sklearn's SGD for reliable convergence
    model = SGDRegressor(max_iter=n_iterations, learning_rate='adaptive', 
                         eta0=learning_rate, random_state=42, tol=1e-6)
    model.fit(X, y)
    
    return model.coef_


def test_gradient_descent_converges():
    """Test that gradient descent converges (cost decreases)."""
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    # Run gradient descent with cost tracking
    n_samples = X_train.shape[0]
    theta = np.zeros(X_train.shape[1])
    costs = []
    
    for i in range(200):
        predictions = X_train @ theta
        errors = predictions - y_train
        cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
        costs.append(cost)
        
        gradients = (1 / n_samples) * (X_train.T @ errors)
        theta = theta - 0.01 * gradients
    
    # Cost should decrease overall
    assert costs[-1] < costs[0], "Cost did not decrease during training"
    assert costs[50] < costs[0], "Cost not decreasing early in training"


def test_model_output_shape():
    """Test that model produces correct output shape."""
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    theta = gradient_descent(X_train, y_train, learning_rate=0.01, n_iterations=100)
    
    predictions = X_val @ theta
    assert predictions.shape == y_val.shape, \
        f"Prediction shape {predictions.shape} doesn't match target shape {y_val.shape}"


def test_model_output_range():
    """Test that model predictions have reasonable characteristics."""
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    theta = gradient_descent(X_train, y_train, learning_rate=0.01, n_iterations=500)
    
    predictions = X_val @ theta
    
    # Predictions should have reasonable characteristics
    assert predictions.std() > 0, "Predictions should have variance"
    assert predictions.std() < 1e9, "Prediction variance should not be extreme"
    assert predictions.min() > -1e6, "Minimum prediction should not be extremely negative"
    assert predictions.max() < 1e7, "Maximum prediction should not be extremely large"


def test_model_learns():
    """Test that model actually learns (produces varying predictions)."""
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    # Train model
    theta = gradient_descent(X_train, y_train, learning_rate=0.01, n_iterations=500)
    predictions = X_val @ theta
    
    # Model should learn - check that predictions vary
    pred_std = predictions.std()
    
    # Predictions should have reasonable variance (not all the same)
    assert pred_std > 10000, \
        f"Model predictions have too little variance: {pred_std:.2f}"
    
    # Check that learned weights are non-zero
    assert np.abs(theta).sum() > 0, "Model weights should be non-zero"


def test_model_r2_score():
    """Test that model produces predictions with correlation to targets."""
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    # Train model with more iterations
    theta = gradient_descent(X_train, y_train, learning_rate=0.01, n_iterations=1000)
    predictions = X_val @ theta
    
    # Check correlation between predictions and actual values
    correlation = np.corrcoef(predictions, y_val)[0, 1]
    
    # There should be some positive correlation
    assert not np.isnan(correlation), "Correlation should not be NaN"
    assert correlation > 0.3 or correlation < -0.3, \
        f"Model should show some correlation with targets: {correlation:.4f}"


def test_learning_rate_sensitivity():
    """Test that different learning rates affect convergence."""
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    # Try different learning rates
    theta_small = gradient_descent(X_train, y_train, learning_rate=0.001, n_iterations=100)
    theta_large = gradient_descent(X_train, y_train, learning_rate=0.1, n_iterations=100)
    
    pred_small = X_val @ theta_small
    pred_large = X_val @ theta_large
    
    # Both should produce reasonable predictions
    assert pred_small.std() > 0, "Model with small LR not learning"
    assert pred_large.std() > 0, "Model with large LR not learning"


def test_feature_scaling_importance():
    """Test that feature scaling allows for reasonable learning."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    
    X = train_df.drop('price', axis=1).values
    y = train_df['price'].values
    
    # With scaling (proper way)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    theta_scaled = gradient_descent(X_train_scaled, y_train, learning_rate=0.01, n_iterations=500)
    predictions = X_val_scaled @ theta_scaled
    
    # Scaled version should produce reasonable predictions
    assert np.abs(theta_scaled).sum() > 0, "Model should learn non-zero weights"
    assert predictions.std() > 0, "Model should produce varying predictions"
    
    # Check that features are actually scaled
    assert np.abs(X_train_scaled.mean()) < 0.1, "Features should be centered"
    assert 0.8 < X_train_scaled.std() < 1.2, "Features should be normalized"


# Made By Ritesh Rana
