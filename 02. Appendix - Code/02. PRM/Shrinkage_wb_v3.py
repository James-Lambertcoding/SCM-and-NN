# Bring in the package
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.datasets import make_hastie_10_2
from matplotlib import pyplot as plt

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# Importing the data
df = pd.read_csv('Wb Data/PRM_v3_norm.csv', index_col=0)

# print(df.columns)

# # Define the predictor
X = df.iloc[:,1:28]


# # Define the target variable
y = df['Australia']

# # Split the whole data in train vs test data. We take a third of the data as testing sample.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
#
# # Define and estimate the model
lr = LinearRegression().fit(X_train, y_train)
ypred_lr = lr.predict(X_test)

# Let's calculate now the forecasting performance based on the $R^2$ coefficient.
r2_lr = round(r2_score(y_test, ypred_lr), 2)
accuracy = {'ols': {'r2 ': r2_lr}}

# ----------------------------------------------------------
# Estimate a ridge regression model
# We now estimate a ridge regression model with a fixed penalty parameter.
# Notice in the python syntax, the penalty term   is called alpha


# This is the penalty term
alpha = 10

# Normalise means that the predictors are
ridge = Ridge(alpha=alpha, fit_intercept=True, max_iter=50000)

ridge.fit(X_train, y_train)
ypred_ridge = ridge.predict(X_test)

r2_ridge = round(r2_score(y_test, ypred_ridge), 2)
accuracy['ridge'] = r2_ridge

# ---------------------------------------------------------------------------
# Estimate a lasso regression
# We now estimate a ridge regression model with a fixed penalty parameter.
# Notice in the python syntax, the penalty term $\lambda$ is called ***alpha***.
# For simplicity, we use the same penalty term used for the ridge regression.


# This is the penalty term
alpha = 10
# Normalise means that the predictors are
lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=50000)

lasso.fit(X_train, y_train)

ypred_lasso = lasso.predict(X_test)

r2_lasso = round(r2_score(y_test, ypred_lasso), 2)

accuracy['lasso'] = r2_lasso

# We can now collect the results across models and compare the performances.
accuracy = pd.DataFrame(accuracy)


# For this level of shrinkage, $\lambda=10$ the OLS seems to perform better than the others.
# We now try to estimate the penalty terms, instead of fixing them, via cross-validation.
#
# ------------------------------------------------------------------
# Cross-validation of the penalty terms
# Cross validation of the penalty terms can be implemented via the function ***GridSearchCV***,
# both for the ridge and for the lasso regression models.


# Here we explore a penalty term from 0.1 to 100
alpha_space = np.linspace(0.1, 100, 20)
param_grid = {'alpha': alpha_space}

ridge_cv = GridSearchCV(Ridge(), param_grid, cv=5)

# We can now fit the model and produce the corresponding forecasts
ridge_cv.fit(X_train, y_train)
ypred_ridge_cv = ridge_cv.predict(X_test)
r2_ridge_cv = round(r2_score(y_test, ypred_ridge_cv), 2)
accuracy['ridge cv'] = r2_ridge_cv

# We can use the same procedure for the lasso.
lasso_cv  = GridSearchCV(Lasso(), param_grid, cv=5)
lasso_cv.fit(X_train, y_train)
ypred_lasso_cv = lasso_cv.predict(X_test)
r2_lasso_cv = round(r2_score(y_test, ypred_lasso_cv), 2)

accuracy['lasso cv'] = r2_lasso_cv


print('Penalty term for the ridge regression: \n', ridge_cv.best_estimator_)
print('Penalty term for the lasso regression: \n', lasso_cv.best_estimator_)

#-----------------

# This is the penalty term
alpha_ridge_cv = ridge_cv.best_params_['alpha']
alpha_lasso_cv = lasso_cv.best_params_['alpha']


print(alpha_ridge_cv)
print(alpha_lasso_cv)

# Normalise means that the predictors are
ridge_df = Ridge(alpha=alpha_ridge_cv, fit_intercept=True)

ridge_df.fit(X_train, y_train)
ypred_ridge_df = ridge_df.predict(X_test)

r2_ridge_df = round(r2_score(y_test, ypred_ridge_df), 2)
accuracy['ridge_df'] = r2_ridge_df

print(accuracy)

## run lasso
lasso_df = Lasso(alpha=alpha_lasso_cv, fit_intercept=True)

lasso_df.fit(X_train, y_train)
ypred_lasso_df = lasso_df.predict(X_test)

r2_lasso_df = round(r2_score(y_test, ypred_ridge_df), 2)
accuracy['lasso_df'] = r2_ridge_df




with open('Outputs/coef_out_ridge_cv_v3_norm.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(X.columns)
    writer.writerow(ridge_df.coef_)

with open('Outputs/coef_out_lasso_cv_v3_norm.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(X.columns)
    writer.writerow(lasso.coef_)





