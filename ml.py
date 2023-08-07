import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('train.csv')

# Split features and target
X = data.drop('target_column', axis=1)
y = data['target_column']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = [
    ('Linear Regression', LinearRegression(), {'normalize': [True, False]}),
    ('Random Forest', RandomForestRegressor(), {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}),
    ('Gradient Boosting', GradientBoostingRegressor(), {'n_estimators': [50, 100, 150], 'max_depth': [3, 4, 5]})
]

# Iterate through models
for name, model, param_grid in models:
    print(f"=== {name} ===")
    
    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Best hyperparameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print("Cross-Validation Scores:", cv_scores)
    
    # Fit the best model on the entire training set
    best_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate RMSE on test set
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Test RMSE:", rmse)
    
    print("===================")
