# Data Dictionary - Email Spam Detection Dataset

## Overview
This dataset contains features extracted from emails for binary classification (spam vs. not spam). It's designed for practicing classification algorithms including Logistic Regression, Gaussian Discriminant Analysis (GDA), and Naive Bayes.

## Files
- `train.csv`: Training dataset with 3,200 samples
- `test.csv`: Test dataset with 800 samples
- `PS2-data.zip`: Original compressed dataset (legacy format)

## Target Variable
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| **spam** | int | Email classification | 0 = not spam (ham), 1 = spam |

## Features

### Word Frequency Features (20 features)
Frequency of common spam-indicative words in the email:

| Feature | Description | Range |
|---------|-------------|-------|
| word_free_freq | Frequency of "free" | 0.0 - 15.0 |
| word_money_freq | Frequency of "money" | 0.0 - 15.0 |
| word_win_freq | Frequency of "win" | 0.0 - 15.0 |
| word_click_freq | Frequency of "click" | 0.0 - 15.0 |
| word_buy_freq | Frequency of "buy" | 0.0 - 15.0 |
| word_discount_freq | Frequency of "discount" | 0.0 - 15.0 |
| word_offer_freq | Frequency of "offer" | 0.0 - 15.0 |
| word_sale_freq | Frequency of "sale" | 0.0 - 15.0 |
| word_price_freq | Frequency of "price" | 0.0 - 15.0 |
| word_save_freq | Frequency of "save" | 0.0 - 15.0 |
| word_meeting_freq | Frequency of "meeting" | 0.0 - 5.0 |
| word_schedule_freq | Frequency of "schedule" | 0.0 - 5.0 |
| word_project_freq | Frequency of "project" | 0.0 - 5.0 |
| word_report_freq | Frequency of "report" | 0.0 - 5.0 |
| word_update_freq | Frequency of "update" | 0.0 - 5.0 |
| word_team_freq | Frequency of "team" | 0.0 - 5.0 |
| word_work_freq | Frequency of "work" | 0.0 - 5.0 |
| word_email_freq | Frequency of "email" | 0.0 - 5.0 |
| word_please_freq | Frequency of "please" | 0.0 - 5.0 |
| word_thanks_freq | Frequency of "thanks" | 0.0 - 5.0 |

### Character Frequency Features (5 features)
Frequency of special characters:

| Feature | Description | Range |
|---------|-------------|-------|
| char_!_freq | Frequency of '!' | 0.0 - 10.0 |
| char_$_freq | Frequency of '$' | 0.0 - 10.0 |
| char_#_freq | Frequency of '#' | 0.0 - 10.0 |
| char_@_freq | Frequency of '@' | 0.0 - 10.0 |
| char_%_freq | Frequency of '%' | 0.0 - 10.0 |

### Additional Features (25 features)
| Feature | Description | Range |
|---------|-------------|-------|
| feature_0 to feature_24 | Additional email characteristics | 0.0 - 10.0 |

**Total Features**: 50

## Data Characteristics

### Class Distribution
- **Training Set**: ~35% spam, ~65% not spam (imbalanced)
- **Test Set**: Similar distribution to training set

### Missing Values
- No missing values in this dataset

### Feature Scaling
- Features have exponential distribution
- Spam emails tend to have higher word frequencies
- Feature scaling recommended for some algorithms

### Class Separability
- Spam and ham classes have overlapping but distinguishable distributions
- Some features are more discriminative than others

## Data Collection Context
This is a synthetic dataset inspired by real spam detection systems like:
- Gmail spam filter
- Microsoft Outlook junk mail filter
- SpamAssassin

## Use Cases
This dataset is ideal for:
1. **Binary Classification**: Spam vs. not spam
2. **Logistic Regression**: Discriminative modeling
3. **Gaussian Discriminant Analysis**: Generative modeling
4. **Naive Bayes**: Probabilistic classification with independence assumption
5. **Model Comparison**: Comparing generative vs discriminative approaches

## Expected Model Performance

### Baseline (Random/Most Frequent)
- Accuracy: 65% (always predict ham)
- Precision/Recall: Poor

### Logistic Regression
- Accuracy: 93-95%
- ROC AUC: 0.96-0.98
- Good balance of precision and recall

### Gaussian Discriminant Analysis
- Accuracy: 90-92%
- ROC AUC: 0.93-0.95
- Assumes Gaussian distribution (may not be perfect fit)

### Naive Bayes
- Accuracy: 91-93%
- ROC AUC: 0.94-0.96
- Fast, works well despite independence assumption

## Business Context

### Real-World Application
Email spam filtering is critical for:
- User experience (inbox cleanliness)
- Security (phishing, malware)
- Productivity (time saved)

### Stakeholders
- **Email Users**: Want clean inbox with no false positives
- **IT Security**: Protect against malicious emails
- **Email Providers**: Reduce server load, improve UX
- **Businesses**: Prevent phishing and data breaches

### Cost of Errors
- **False Positive (Ham → Spam)**: Important email missed, HIGH COST
- **False Negative (Spam → Ham)**: User annoyed, LOWER COST
- **Decision**: Prefer to err on side of letting spam through

## Key Challenges

1. **Class Imbalance**: More ham than spam
2. **Feature Independence**: Naive Bayes assumption may be violated
3. **Threshold Tuning**: Adjust for business costs
4. **Evolving Spam**: Real spam patterns change over time

## Feature Engineering Ideas

### Feature Transformations
```python
# Log transformation for exponential features
log_features = np.log1p(features)

# Binary indicators
has_dollar_sign = (char_$_freq > 0).astype(int)

# Feature combinations
spam_word_count = word_free_freq + word_money_freq + word_win_freq
```

### Feature Selection
```python
# High spam correlation
spam_indicators = ['word_free_freq', 'word_money_freq', 'char_$_freq']

# Ham indicators
ham_indicators = ['word_meeting_freq', 'word_project_freq']
```

## Evaluation Metrics

### Primary Metrics
1. **ROC AUC**: Overall discriminative ability
2. **Precision**: Of flagged spam, how much is actually spam?
3. **Recall**: Of actual spam, how much did we catch?
4. **F1 Score**: Balance of precision and recall

### Business Metrics
- **False Positive Rate**: Must be < 1% (important emails missed)
- **True Positive Rate**: Should be > 90% (spam caught)

## Known Limitations

1. **Simplified Features**: Real systems use thousands of features
2. **No Content**: Actual email text not included
3. **Static Dataset**: Real spam evolves continuously
4. **No Sender Info**: Domain reputation not included
5. **No User Behavior**: No user feedback signals

## License
Dataset created for educational purposes as part of the Machine Learning Course.

---

**Made By Ritesh Rana**
