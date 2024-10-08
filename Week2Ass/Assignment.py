import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# id:2-2-2

df = pd.read_csv('week2.csv')
print(df.head())
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

#(a)
plt.figure(figsize=(8, 6))

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X, y)
#training data
#  points where the target is +1
plt.scatter(X1[y == 1], X2[y == 1], marker='x', color='green', label='+1 target')

#  points where the target is -1
plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='-1 target')

plt.xlabel('X_1')
plt.ylabel('X_2')
plt.title('Data Visualisation')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X, y)
#training data
#  points where the target is +1
plt.scatter(X1[y == 1], X2[y == 1], marker='x', color='green', label='+1 target')

#  points where the target is -1
plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='-1 target')

predictions = logistic_regression_model.predict(X)

#predictions from regression model
plt.scatter(X1[predictions == 1], X2[predictions == 1], marker='v',
            color='orange', label='predicted +1', alpha=0.4)
plt.scatter(X1[predictions == -1], X2[predictions == -1], marker='P',
            color='red', label='predicted -1', alpha=0.5)

coefficients = logistic_regression_model.coef_[0]
intercept = logistic_regression_model.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')
# Which feature has the most influence
# if abs(coefficients[0][0]) > abs(coefficients[0][1]):
#     most_influential = "Feature 1 (X1)"
# else:
#     most_influential = "Feature 2 (X2)"
#
# print(f'The most influential feature is: {most_influential}')


x1_vals = np.linspace(X1.min(), X1.max(), 200)
x2_vals = -(coefficients[0] * x1_vals + intercept) / coefficients[1]
plt.plot(x1_vals, x2_vals, color='black', linestyle='--', linewidth=1,
         label='Decision Boundary')

plt.xlabel('X_1')
plt.ylabel('X_2')
plt.title('Logistic Regression: Training Data, Predictions, and Decision Boundary')
plt.legend()
plt.show()

# (b)
C_values = [0.001, 1, 100]

svm_models = {}

for C in C_values:
    svm_model = LinearSVC(C=C, max_iter=10000)
    svm_model.fit(X, y)
    svm_models[C] = svm_model
    print(f"Model parameters for C = {C}: Coefficients: {svm_model.coef_[0]},"
          f" Intercept: {svm_model.intercept_[0]}")

for C in C_values:
    svm_model = svm_models[C]

    # Predictions
    predictions = svm_model.predict(X)
    # Create new plot for each C
    plt.figure(figsize=(8, 6))

    # Plot training data
    plt.scatter(X1[y == 1], X2[y == 1], marker='x', color='green', label='+1 target')
    plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='-1 target')

    # Plot predictions
    plt.scatter(X1[predictions == 1], X2[predictions == 1], marker='v', color='orange',
                label='Predicted +1', alpha=0.4)
    plt.scatter(X1[predictions == -1], X2[predictions == -1], marker='P', color='red',
                label='Predicted -1', alpha=0.5)

    # Calculate and plot decision boundary
    coefficients = svm_model.coef_[0]
    intercept = svm_model.intercept_[0]
    x1_vals = np.linspace(X1.min(), X1.max(), 200)
    x2_vals = -(coefficients[0] * x1_vals + intercept) / coefficients[1]

    plt.plot(x1_vals, x2_vals, color='black', linestyle='--', linewidth=1,
             label=f'Decision Boundary (C = {C})')
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.title(f'Linear SVM: Training Data, Predictions, and Decision Boundary (C = {C})')
    plt.legend()
    plt.show()

# (c)
# Adding new features: squares of the original features
X1_squared = X1 ** 2
X2_squared = X2 ** 2

X_new = np.column_stack((X1, X2, X1_squared, X2_squared))

logistic_regression_model_new = LogisticRegression()
logistic_regression_model_new.fit(X_new, y)

# Get  model parameters
coefficients_new = logistic_regression_model_new.coef_[0]
intercept_new = logistic_regression_model_new.intercept_[0]
print("Part C: Model and Trained Parameter Values")
print(f"Coefficients: {coefficients_new}")
print(f"Intercept: {intercept_new}")

predictions_new = logistic_regression_model_new.predict(X_new)

plt.figure(figsize=(8, 6))

plt.scatter(X1[y == 1], X2[y == 1], marker='x', color='green', label='+1 target')
plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='-1 target')

plt.scatter(X1[predictions_new == 1], X2[predictions_new == 1], marker='v',
            color='orange', label='Predicted +1',
            alpha=0.4)
plt.scatter(X1[predictions_new == -1], X2[predictions_new == -1], marker='P',
            color='red', label='Predicted -1',
            alpha=0.5)

plt.xlabel('X_1')
plt.ylabel('X_2')
plt.title('Logistic Regression with Quadratic Features: Predictions and Training Data')
plt.legend()
plt.show()

# (c)(iii) Performance Comparison Against a Baseline Predictor
y_mapped = np.where(y == -1, 0, 1)

# Baseline predictor: Always predicts the most common class
most_common_class = np.bincount(y_mapped).argmax()

# Map the most common class back to original values (-1 or 1)
baseline_prediction = np.where(most_common_class == 0, -1, 1)

new_model_accuracy = np.mean(predictions_new == y)
baseline_accuracy = np.mean(baseline_prediction == y)

print(f"New Model Accuracy: {new_model_accuracy}")
print(f"Baseline Accuracy: {baseline_accuracy}")