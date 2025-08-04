# Feature Engineering and Data Cleaning

This project demonstrates fundamental feature engineering techniques to clean and prepare data for analysis and machine learning.

## Overview

In this repository, I cover:

- Handling missing values (NA) using mean, median, and mode.
- Visualizing data distributions and detecting outliers with box plots.
- Using the Interquartile Range (IQR) method for outlier detection.
- Calculating summary statistics like mean, median, and mode.
- Basic data cleaning steps to improve dataset quality.

## Example Code Snippets

### Filling Missing Values

python
import pandas as pd

# Sample data
data = {'Age': [25, 30, None, 35, None, 40]}
df = pd.DataFrame(data)

# Fill NA with mean
df['Age_mean'] = df['Age'].fillna(df['Age'].mean())

# Fill NA with median
df['Age_median'] = df['Age'].fillna(df['Age'].median())

# Fill NA with mode
df['Age_mode'] = df['Age'].fillna(df['Age'].mode()[0])

print(df)
