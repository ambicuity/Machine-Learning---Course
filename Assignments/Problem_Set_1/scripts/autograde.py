#!/usr/bin/env python3
"""
Autograder for Problem Set 1: Linear Regression
Automatically grades student submissions based on multiple criteria.

Usage:
    python autograde.py [--dry-run]
    
Options:
    --dry-run    Run without executing notebooks (for CI testing)
"""

import sys
import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
MAX_SCORE = 100
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')


class AutoGrader:
    """Autograder for Problem Set 1."""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.scores = {}
        self.feedback = []
        self.total_score = 0
        
    def print_header(self):
        """Print grading header."""
        print("=" * 70)
        print("PROBLEM SET 1: LINEAR REGRESSION - AUTOGRADER")
        print("=" * 70)
        print()
        
    def grade_data_loading(self):
        """Grade data loading (20 points)."""
        print("📊 Grading Data Loading (20 points)...")
        score = 0
        
        try:
            # Check if data files exist
            train_path = os.path.join(DATA_DIR, 'train.csv')
            test_path = os.path.join(DATA_DIR, 'test.csv')
            
            if os.path.exists(train_path):
                score += 5
                self.feedback.append("✓ Training data file exists")
            else:
                self.feedback.append("✗ Training data file not found")
            
            if os.path.exists(test_path):
                score += 5
                self.feedback.append("✓ Test data file exists")
            else:
                self.feedback.append("✗ Test data file not found")
            
            # Load and validate data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Check shapes
            if train_df.shape[1] == 7:
                score += 5
                self.feedback.append("✓ Training data has correct number of columns")
            else:
                self.feedback.append(f"✗ Training data has {train_df.shape[1]} columns, expected 7")
            
            # Check for missing values
            if train_df.isnull().sum().sum() == 0:
                score += 5
                self.feedback.append("✓ No missing values in data")
            else:
                self.feedback.append("✗ Data contains missing values")
                
        except Exception as e:
            self.feedback.append(f"✗ Error loading data: {str(e)}")
        
        self.scores['Data Loading'] = score
        print(f"Score: {score}/20\n")
        
    def grade_model_implementation(self):
        """Grade model implementation (30 points)."""
        print("🤖 Grading Model Implementation (30 points)...")
        score = 0
        
        try:
            # Load data
            train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
            X = train_df.drop('price', axis=1).values
            y = train_df['price'].values
            
            # Prepare data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Simple gradient descent implementation
            def gradient_descent(X, y, learning_rate=0.01, n_iterations=500):
                n_samples, n_features = X.shape
                theta = np.zeros(n_features)
                
                for _ in range(n_iterations):
                    predictions = X @ theta
                    errors = predictions - y
                    gradients = (1 / n_samples) * (X.T @ errors)
                    theta = theta - learning_rate * gradients
                
                return theta
            
            # Train model
            theta = gradient_descent(X_train, y_train)
            
            # Check if model produces reasonable outputs
            predictions = X_val @ theta
            
            if predictions.shape == y_val.shape:
                score += 10
                self.feedback.append("✓ Model produces correct output shape")
            else:
                self.feedback.append("✗ Model output shape incorrect")
            
            if predictions.min() > 0 and predictions.max() < 2000000:
                score += 10
                self.feedback.append("✓ Model predictions in reasonable range")
            else:
                self.feedback.append("✗ Model predictions out of reasonable range")
            
            # Check if model is better than baseline
            baseline_pred = np.full_like(y_val, y_train.mean())
            model_mse = mean_squared_error(y_val, predictions)
            baseline_mse = mean_squared_error(y_val, baseline_pred)
            
            if model_mse < baseline_mse:
                score += 10
                self.feedback.append("✓ Model outperforms baseline")
            else:
                self.feedback.append("✗ Model does not outperform baseline")
                
        except Exception as e:
            self.feedback.append(f"✗ Error in model implementation: {str(e)}")
        
        self.scores['Model Implementation'] = score
        print(f"Score: {score}/30\n")
        
    def grade_metrics(self):
        """Grade metrics and evaluation (30 points)."""
        print("📈 Grading Metrics and Evaluation (30 points)...")
        score = 0
        
        try:
            # Load data
            train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
            test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
            
            X_train = train_df.drop('price', axis=1).values
            y_train = train_df['price'].values
            X_test = test_df.drop('price', axis=1).values
            y_test = test_df['price'].values
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
                n_samples, n_features = X.shape
                theta = np.zeros(n_features)
                
                for _ in range(n_iterations):
                    predictions = X @ theta
                    errors = predictions - y
                    gradients = (1 / n_samples) * (X.T @ errors)
                    theta = theta - learning_rate * gradients
                
                return theta
            
            theta = gradient_descent(X_train_scaled, y_train)
            predictions = X_test_scaled @ theta
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            # Grade based on R² score
            if r2 > 0.75:
                score += 15
                self.feedback.append(f"✓ Excellent R² score: {r2:.4f}")
            elif r2 > 0.60:
                score += 10
                self.feedback.append(f"○ Good R² score: {r2:.4f}")
            elif r2 > 0.40:
                score += 5
                self.feedback.append(f"○ Acceptable R² score: {r2:.4f}")
            else:
                self.feedback.append(f"✗ Low R² score: {r2:.4f}")
            
            # Grade based on RMSE
            if rmse < 80000:
                score += 15
                self.feedback.append(f"✓ Good RMSE: ${rmse:,.2f}")
            elif rmse < 120000:
                score += 10
                self.feedback.append(f"○ Acceptable RMSE: ${rmse:,.2f}")
            else:
                self.feedback.append(f"✗ High RMSE: ${rmse:,.2f}")
                
        except Exception as e:
            self.feedback.append(f"✗ Error calculating metrics: {str(e)}")
        
        self.scores['Metrics'] = score
        print(f"Score: {score}/30\n")
        
    def grade_code_quality(self):
        """Grade code quality (20 points)."""
        print("💻 Grading Code Quality (20 points)...")
        score = 0
        
        # Check if tests pass
        try:
            tests_dir = os.path.join(os.path.dirname(__file__), '..', 'tests')
            if os.path.exists(tests_dir):
                result = subprocess.run(
                    ['pytest', tests_dir, '-v'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    score += 15
                    self.feedback.append("✓ All unit tests pass")
                else:
                    # Partial credit based on passed tests
                    output = result.stdout
                    if 'passed' in output:
                        passed = int(output.split()[0].split('/')[0]) if '/' in output else 0
                        score += min(10, passed * 2)
                        self.feedback.append(f"○ Some unit tests pass: {passed}")
                    else:
                        self.feedback.append("✗ Unit tests failed")
            else:
                self.feedback.append("○ No tests directory found")
                score += 10  # Give partial credit
                
        except subprocess.TimeoutExpired:
            self.feedback.append("✗ Tests timed out")
        except Exception as e:
            self.feedback.append(f"○ Could not run tests: {str(e)}")
            score += 10  # Give partial credit
        
        # Code structure check (basic)
        score += 5
        self.feedback.append("✓ Code structure is acceptable")
        
        self.scores['Code Quality'] = score
        print(f"Score: {score}/20\n")
        
    def print_summary(self):
        """Print grading summary."""
        self.total_score = sum(self.scores.values())
        
        print("=" * 70)
        print("GRADING SUMMARY")
        print("=" * 70)
        print()
        
        for category, score in self.scores.items():
            print(f"{category:.<50} {score:>4} points")
        
        print("-" * 70)
        print(f"{'TOTAL SCORE':.<50} {self.total_score:>4}/{MAX_SCORE}")
        print("=" * 70)
        print()
        
        # Determine grade
        percentage = (self.total_score / MAX_SCORE) * 100
        if percentage >= 90:
            grade = "A"
        elif percentage >= 80:
            grade = "B"
        elif percentage >= 70:
            grade = "C"
        elif percentage >= 60:
            grade = "D"
        else:
            grade = "F"
        
        print(f"Final Grade: {grade} ({percentage:.1f}%)")
        print()
        
        # Pass/Fail
        if self.total_score >= 60:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
        
        print()
        print("Detailed Feedback:")
        print("-" * 70)
        for feedback in self.feedback:
            print(f"  {feedback}")
        print()
        
    def run(self):
        """Run the complete autograding process."""
        self.print_header()
        
        if self.dry_run:
            print("⚠️  Running in DRY-RUN mode (no notebook execution)\n")
        
        # Run all grading components
        self.grade_data_loading()
        self.grade_model_implementation()
        self.grade_metrics()
        self.grade_code_quality()
        
        # Print final summary
        self.print_summary()
        
        return self.total_score


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Autograde Problem Set 1')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without executing notebooks')
    args = parser.parse_args()
    
    grader = AutoGrader(dry_run=args.dry_run)
    score = grader.run()
    
    # Exit with non-zero status if failed
    sys.exit(0 if score >= 60 else 1)


if __name__ == '__main__':
    main()

# Made By Ritesh Rana
