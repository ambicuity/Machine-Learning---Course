"""
Unit tests for data loading in Problem Set 1.
Tests data integrity, shape, and basic statistics.
"""

import pytest
import pandas as pd
import numpy as np
import os

# Get the path to the Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')


def test_train_data_exists():
    """Test that training data file exists."""
    train_path = os.path.join(DATA_DIR, 'train.csv')
    assert os.path.exists(train_path), "Training data file not found"


def test_test_data_exists():
    """Test that test data file exists."""
    test_path = os.path.join(DATA_DIR, 'test.csv')
    assert os.path.exists(test_path), "Test data file not found"


def test_train_data_shape():
    """Test that training data has correct shape."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    assert train_df.shape[0] > 0, "Training data is empty"
    assert train_df.shape[1] == 7, f"Expected 7 columns, got {train_df.shape[1]}"


def test_test_data_shape():
    """Test that test data has correct shape."""
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    assert test_df.shape[0] > 0, "Test data is empty"
    assert test_df.shape[1] == 7, f"Expected 7 columns, got {test_df.shape[1]}"


def test_required_columns():
    """Test that all required columns are present."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    required_columns = [
        'square_feet', 'bedrooms', 'bathrooms', 
        'lot_size', 'year_built', 'garage_spaces', 'price'
    ]
    for col in required_columns:
        assert col in train_df.columns, f"Missing required column: {col}"


def test_no_missing_values():
    """Test that there are no missing values."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    missing = train_df.isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values in training data"


def test_target_variable_range():
    """Test that target variable (price) is in reasonable range."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    prices = train_df['price']
    
    assert prices.min() > 0, "Price should be positive"
    assert prices.min() >= 100000, "Minimum price seems too low"
    assert prices.max() <= 2000000, "Maximum price seems too high"


def test_feature_ranges():
    """Test that features are in reasonable ranges."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    
    # Square feet
    assert train_df['square_feet'].min() >= 500, "Square feet too small"
    assert train_df['square_feet'].max() <= 10000, "Square feet too large"
    
    # Bedrooms
    assert train_df['bedrooms'].min() >= 1, "Must have at least 1 bedroom"
    assert train_df['bedrooms'].max() <= 10, "Too many bedrooms"
    
    # Year built
    assert train_df['year_built'].min() >= 1900, "Year built too old"
    assert train_df['year_built'].max() <= 2024, "Year built in the future"


def test_data_types():
    """Test that columns have correct data types."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    
    # Numeric columns
    numeric_cols = ['square_feet', 'bedrooms', 'lot_size', 
                    'year_built', 'garage_spaces', 'price']
    for col in numeric_cols:
        assert np.issubdtype(train_df[col].dtype, np.number), \
            f"{col} should be numeric"


def test_train_test_distribution_similarity():
    """Test that train and test sets have similar distributions."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    # Compare price means (should be within 20%)
    train_mean_price = train_df['price'].mean()
    test_mean_price = test_df['price'].mean()
    
    ratio = test_mean_price / train_mean_price
    assert 0.8 <= ratio <= 1.2, \
        f"Train/test price distributions too different: {ratio:.2f}"


# Made By Ritesh Rana
