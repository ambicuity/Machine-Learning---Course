# Data Dictionary - Housing Prices Dataset

## Overview
This dataset contains information about residential properties and their sale prices. It's designed for practicing linear regression and understanding the relationship between house features and market value.

## Files
- `train.csv`: Training dataset with 800 samples
- `test.csv`: Test dataset with 126 samples  
- `PS1-data.zip`: Original compressed dataset (legacy format)

## Target Variable
| Column | Type | Description | Range |
|--------|------|-------------|-------|
| **price** | float | Sale price of the house in USD | $169,900 - $1,049,900 |

## Features

### Continuous Features

| Column | Type | Description | Units | Range |
|--------|------|-------------|-------|-------|
| **square_feet** | int | Total interior living area | sq ft | 852 - 5,000 |
| **lot_size** | int | Size of the property lot | sq ft | 3,500 - 17,200 |

### Discrete Features

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| **bedrooms** | int | Number of bedrooms | 1 - 5 |
| **bathrooms** | float | Number of bathrooms (includes half baths) | 1.0 - 4.0 |
| **garage_spaces** | int | Number of garage parking spaces | 1 - 3 |
| **year_built** | int | Year of original construction | 1966 - 1999 |

## Data Characteristics

### Missing Values
- No missing values in this cleaned version
- Original dataset may have had missing values

### Outliers
- Dataset includes natural outliers (e.g., luxury homes with 5 bedrooms)
- All values are realistic for the housing market

### Feature Correlations
Expected strong correlations:
- `square_feet` with `price` (positive, strong)
- `bedrooms` with `square_feet` (positive, moderate)
- `year_built` with `price` (positive, weak)
- `garage_spaces` with `price` (positive, moderate)

### Scale Differences
Features have very different scales:
- `square_feet`: hundreds to thousands
- `year_built`: 1960s to 1990s
- `bedrooms`: single digits
- `price`: hundreds of thousands

⚠️ **Important**: Feature scaling is essential for gradient descent!

## Data Collection
This is a synthetic dataset inspired by real housing market data. It simulates:
- Mid-sized US city housing market
- Properties built between 1966-1999
- Typical single-family homes
- Realistic price-to-feature relationships

## Use Cases
This dataset is ideal for:
1. **Linear Regression**: Predicting house prices
2. **Feature Engineering**: Creating polynomial/interaction features
3. **Gradient Descent**: Learning optimization from scratch
4. **Model Evaluation**: Comparing train vs test performance
5. **Data Preprocessing**: Normalization, standardization

## Expected Model Performance

### Baseline (Mean Prediction)
- R² Score: 0.00
- RMSE: ~$150,000

### Linear Regression (Well-Tuned)
- R² Score: 0.75 - 0.85
- RMSE: $50,000 - $80,000

### With Feature Engineering
- R² Score: 0.80 - 0.90
- RMSE: $40,000 - $70,000

## Business Context

### Real-World Application
Similar to Zillow's "Zestimate" or Redfin's home value estimates, this model predicts property values based on observable characteristics.

### Stakeholders
- **Home Buyers**: Understanding fair market value
- **Real Estate Agents**: Pricing recommendations
- **Banks/Lenders**: Property valuation for mortgages
- **Investors**: Identifying undervalued properties

### Cost of Errors
- **Overestimation**: Buyers pay too much, reduced sales
- **Underestimation**: Sellers lose money, opportunity cost
- Typical acceptable error: ±5-10% of actual price

## Feature Engineering Ideas

### Polynomial Features
```python
# Age of house
age = current_year - year_built

# Price per square foot
price_per_sqft = price / square_feet

# Square footage squared
sqft_squared = square_feet ** 2
```

### Interaction Features
```python
# Bedroom-bathroom ratio
bed_bath_ratio = bedrooms / bathrooms

# Living space quality
quality_score = square_feet * (bathrooms / bedrooms)

# Land value proxy
land_value = lot_size * garage_spaces
```

### Categorical Encoding
```python
# Age categories
age_group = 'new' if age < 10 else 'mid' if age < 25 else 'old'

# Size categories
size_category = 'small' if square_feet < 1500 else 'medium' if square_feet < 2500 else 'large'
```

## Known Limitations

1. **No Location Data**: Real estate is highly location-dependent
2. **No Condition Info**: House condition affects price significantly
3. **No Market Timing**: Housing prices vary over time
4. **Simplified Features**: Real datasets have 50+ features
5. **No Amenities**: Pool, fireplace, renovations not included

## Citation
Dataset created for educational purposes as part of the Machine Learning Course.

## License
Free to use for educational purposes.

---

**Made By Ritesh Rana**
