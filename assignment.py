import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# id:24--24-24

df = pd.read_csv('week3.csv')

# (i)
# a
X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
plt.figure(figsize=(8,6))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel('Feature 1: X1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
plt.title("id:24--24-24")
plt.show()

X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
plt.figure(figsize=(8,6))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel('Feature 1: X1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.view_init(180)
plt.title("id:24--24-24")
plt.show()

X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
plt.figure(figsize=(8,6))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel('Feature 1: X1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.view_init(360)
plt.title("id:24--24-24")

plt.show()

# b

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

C_values = [1, 5, 10, 50, 100, 500, 1000]
alpha_values = [1 / (2 * C) for C in C_values]

# Train Lasso models for different values of C
for alpha, C in zip(alpha_values, C_values):

    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    lasso_model.fit(X_poly, y)

    coefficients = lasso_model.coef_
    intercept = lasso_model.intercept_

    print(f"\nLasso Regression with C={C} (alpha={alpha}):")
    print(f"Intercept: {intercept}")

    for idx, coef in enumerate(coefficients):
        feature_name = poly.get_feature_names_out()[idx]
        print(f"Feature: {feature_name}, Coefficient: {coef}")

# c

grid = np.linspace(-5, 5)
X_test = []

for i in grid:
    for j in grid:
        X_test.append([i, j])

X_test = np.array(X_test)

# polynomial features
X_test_poly = poly.transform(X_test)

for alpha, C in zip(alpha_values, C_values):

    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    lasso_model.fit(X_poly, y)

    y_pred = lasso_model.predict(X_test_poly)

    X0_grid, X1_grid = np.meshgrid(grid, grid)
    Y_pred_grid = y_pred.reshape(X0_grid.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X0_grid, X1_grid, Y_pred_grid,
                    alpha=0.5, cmap='viridis')

    ax.scatter(X[:, 0], X[:, 1], y, color='r',
               label='Training Data')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    ax.set_title(f'Lasso Regression Predictions with C={C}')

    plt.legend()
    plt.show()

# e
C_values = [1, 5, 10, 50, 100, 500, 1000]
alpha_values = [1 / (2 * C) for C in C_values]

# Train Ridge models for different values of C
for alpha, C in zip(alpha_values, C_values):

    ridge_model = Ridge(alpha=alpha, max_iter=10000)
    ridge_model.fit(X_poly, y)

    coefficients = ridge_model.coef_
    intercept = ridge_model.intercept_

    print(f"\nRidge Regression with C={C} (alpha={alpha}):")
    print(f"Intercept: {intercept}")

    for idx, coef in enumerate(coefficients):
        feature_name = poly.get_feature_names_out()[idx]
        print(f"Feature: {feature_name}, Coefficient: {coef}")

    y_pred = ridge_model.predict(X_test_poly)

    X0_grid, X1_grid = np.meshgrid(grid, grid)
    Y_pred_grid = y_pred.reshape(X0_grid.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X0_grid, X1_grid, Y_pred_grid,
                    alpha=0.5, cmap='viridis')

    ax.scatter(X[:, 0], X[:, 1], y, color='r',
               label='Training Data')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    ax.set_title(f'Ridge Regression Predictions with C={C}')

    plt.legend()
    plt.show()

# (ii)
# a

C_values = [0.1, 1, 10, 100, 1000]
alpha_values = [1 / (2 * C) for C in C_values]

mean_errors = []
std_errors = []

for alpha in alpha_values:
    lasso_model = Lasso(alpha=alpha, max_iter=10000)

    scores = cross_val_score(lasso_model, X_poly, y, cv=5,
                             scoring='neg_mean_squared_error')

    mean_errors.append(-scores.mean())
    std_errors.append(scores.std())

plt.figure(figsize=(10, 6))
plt.errorbar(C_values, mean_errors, yerr=std_errors, fmt='o-',
             capsize=5, label='Cross-validation error')
plt.xscale('log')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Mean Squared Error')
plt.title('5-Fold Cross-Validation Error vs C '
          'for Lasso Regression')
plt.legend()
plt.grid(True)
plt.show()

# c

C_values_ridge = [0.1, 1, 10, 100, 1000]
alpha_values_ridge = [1 / (2 * C) for C in C_values_ridge]

mean_errors_ridge = []
std_errors_ridge = []

for alpha in alpha_values_ridge:
    ridge_model = Ridge(alpha=alpha, max_iter=10000)

    scores_ridge = cross_val_score(ridge_model, X_poly,
                                   y, cv=5, scoring='neg_mean_squared_error')

    mean_errors_ridge.append(-scores_ridge.mean())
    std_errors_ridge.append(scores_ridge.std())

plt.figure(figsize=(10, 6))
plt.errorbar(C_values_ridge, mean_errors_ridge,
             yerr=std_errors_ridge, fmt='o-', capsize=5,
             label='Cross-validation error (Ridge)')
plt.xscale('log')
plt.xlabel('C (Regularisation Parameter)')
plt.ylabel('Mean Squared Error')
plt.title('5-Fold Cross-Validation Error vs '
          'C for Ridge Regression')
plt.legend()
plt.grid(True)
plt.show()