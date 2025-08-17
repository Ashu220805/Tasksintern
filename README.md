Projects Overview and Definitions
1. Feature Engineering and Data Cleaning

Definition: Preparing raw data for analysis by fixing inconsistencies, handling missing values, and removing outliers. This step ensures that the data is accurate, complete, and usable for machine learning algorithms.
What you learn:
Handling missing data using statistical imputation: mean, median, and mode.
Detecting outliers with boxplots and the Interquartile Range (IQR) method.
Calculating basic summary statistics to understand the data’s central tendency and spread.
Improving dataset quality through cleaning steps like removing duplicates and correcting errors.
Why clean data matters: models trained on clean data perform better and are more reliable.

2. Linear Regression

Definition: A fundamental regression technique that models the relationship between one or more independent variables and a continuous dependent variable by fitting a linear equation to observed data.
What you learn:
Implementing simple linear regression (one feature) and multiple linear regression (multiple features).
Evaluating regression models using metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² (coefficient of determination).
Interpreting regression coefficients to understand feature impacts.
Visualizing regression lines to see how predictions relate to actual values.
Application: Predicting median house prices based on California Housing dataset features.

3. Logistic Regression (Binary Classification)

Definition: A classification algorithm used to predict a binary outcome (yes/no, true/false) by estimating probabilities using a logistic function (sigmoid).
What you learn:
Training a logistic regression classifier on the Breast Cancer Wisconsin dataset.
Standardizing features for better convergence and model performance.
Understanding the sigmoid function and threshold tuning to balance precision and recall.
Evaluating model performance with confusion matrix, precision, recall, F1-score, and ROC-AUC.
How to adjust the classification threshold and why it matters in practice.

4. Decision Trees and Random Forests

Definition:
Decision Trees: Models that split data into subsets based on feature values to make decisions.
Random Forests: Ensemble of decision trees that improve prediction accuracy and reduce overfitting by averaging results from multiple trees.
What you learn:
Training decision trees and visualizing their structure to understand model decisions.
Controlling model complexity by limiting tree depth to prevent overfitting.
Building random forests and comparing their accuracy to single decision trees.
Interpreting feature importance scores to identify influential variables.
Applying cross-validation for robust performance evaluation.

5. K-Nearest Neighbors (KNN) Classification

Definition: A non-parametric classification algorithm that assigns class labels based on the majority vote of the nearest neighbors in feature space.
What you learn:
Normalizing features to ensure distance metrics are meaningful.
Experimenting with different values of K (number of neighbors) to find the best model.
Evaluating model accuracy and confusion matrix for performance insights.
Visualizing decision boundaries in two dimensions to understand classification regions.
Understanding trade-offs between bias and variance in choosing K.

6. Support Vector Machines (SVM)

Definition: A powerful classifier that finds the hyperplane maximizing the margin between classes, using kernel functions to handle linear and non-linear data separation.
What you learn:
Training SVMs with linear and RBF (Radial Basis Function) kernels for different types of data.
Visualizing decision boundaries in two features for intuitive understanding.
Hyperparameter tuning for C (regularization) and gamma (kernel coefficient) using grid search.
Using cross-validation for unbiased performance evaluation.
Key concepts: margin maximization, kernel trick, overfitting control.

Exploratory Data Analysis (EDA)
Definition: The process of summarizing main characteristics of the dataset often with visual methods. It helps to uncover underlying patterns, spot anomalies, check assumptions, and test hypotheses.
What you learn:

Using histograms, boxplots, and pairplots to visualize distributions and relationships.
Detecting outliers with statistical methods like IQR and understanding their impact.
Checking for missing values and duplicates.
Computing descriptive statistics to summarize datasets effectively.

Tools & Libraries Used

Python 3.x: Primary programming language
pandas: Data manipulation and cleaning
numpy: Numerical operations
matplotlib & seaborn: Visualization

scikit-learn: Machine learning algorithms and evaluation

graphviz: Decision tree visualization
