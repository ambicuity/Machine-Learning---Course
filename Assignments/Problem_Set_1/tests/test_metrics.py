"""
Unit tests for evaluation metrics in Problem Set 1.
Tests MSE, RMSE, and R² calculations.
"""

import pytest
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def test_mse_calculation():
    """Test MSE calculation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    # Calculate MSE manually
    mse_manual = np.mean((y_true - y_pred) ** 2)
    
    # Calculate using sklearn
    mse_sklearn = mean_squared_error(y_true, y_pred)
    
    # Should be approximately equal
    np.testing.assert_almost_equal(mse_manual, mse_sklearn, decimal=6)


def test_rmse_calculation():
    """Test RMSE calculation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    # Calculate RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # RMSE should be positive
    assert rmse > 0, "RMSE should be positive"
    
    # RMSE should be less than or equal to max absolute error
    max_error = np.abs(y_true - y_pred).max()
    assert rmse <= max_error, "RMSE should be <= max absolute error"


def test_r2_score_calculation():
    """Test R² score calculation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    # Calculate R² manually
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2_manual = 1 - (ss_res / ss_tot)
    
    # Calculate using sklearn
    r2_sklearn = r2_score(y_true, y_pred)
    
    # Should be approximately equal
    np.testing.assert_almost_equal(r2_manual, r2_sklearn, decimal=6)


def test_r2_score_range():
    """Test that R² score is in expected range."""
    y_true = np.array([100, 200, 300, 400, 500])
    
    # Perfect predictions
    y_pred_perfect = y_true.copy()
    r2_perfect = r2_score(y_true, y_pred_perfect)
    assert r2_perfect == 1.0, "R² should be 1.0 for perfect predictions"
    
    # Good predictions
    y_pred_good = y_true + np.random.normal(0, 10, len(y_true))
    r2_good = r2_score(y_true, y_pred_good)
    assert 0.8 <= r2_good <= 1.0, f"R² for good predictions should be high: {r2_good:.4f}"
    
    # Baseline (predict mean)
    y_pred_mean = np.full_like(y_true, y_true.mean(), dtype=float)
    r2_mean = r2_score(y_true, y_pred_mean)
    np.testing.assert_almost_equal(r2_mean, 0.0, decimal=6)


def test_mse_properties():
    """Test mathematical properties of MSE."""
    y_true = np.array([100, 200, 300, 400, 500])
    
    # MSE should be 0 for perfect predictions
    y_pred_perfect = y_true.copy()
    mse_perfect = mean_squared_error(y_true, y_pred_perfect)
    assert mse_perfect == 0, "MSE should be 0 for perfect predictions"
    
    # MSE should increase with larger errors
    y_pred_small_error = y_true + 1
    y_pred_large_error = y_true + 10
    
    mse_small = mean_squared_error(y_true, y_pred_small_error)
    mse_large = mean_squared_error(y_true, y_pred_large_error)
    
    assert mse_large > mse_small, "MSE should increase with larger errors"


def test_metrics_with_realistic_data():
    """Test metrics with realistic house price data."""
    # Simulate realistic house prices and predictions
    np.random.seed(42)
    n_samples = 100
    
    # True prices: $200k to $600k
    y_true = np.random.uniform(200000, 600000, n_samples)
    
    # Predictions with ~10% error
    y_pred = y_true + np.random.normal(0, 40000, n_samples)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Check that metrics are in reasonable ranges
    assert rmse > 0, "RMSE should be positive"
    assert rmse < 100000, f"RMSE seems too large: ${rmse:.2f}"
    
    assert 0.5 <= r2 <= 1.0, f"R² should be reasonable: {r2:.4f}"


def test_metrics_consistency():
    """Test that metrics are consistent across different calculations."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    # Calculate MSE in different ways
    mse1 = mean_squared_error(y_true, y_pred)
    mse2 = np.mean((y_true - y_pred) ** 2)
    mse3 = np.sum((y_true - y_pred) ** 2) / len(y_true)
    
    # All should be equal
    np.testing.assert_almost_equal(mse1, mse2, decimal=10)
    np.testing.assert_almost_equal(mse2, mse3, decimal=10)


def test_percentage_error_calculation():
    """Test mean absolute percentage error calculation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # MAPE should be positive
    assert mape > 0, "MAPE should be positive"
    
    # MAPE should be less than 100% for reasonable predictions
    assert mape < 100, f"MAPE seems too large: {mape:.2f}%"


# Made By Ritesh Rana
