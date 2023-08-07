import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import datetime
import joblib

# Read the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Split data into features (X) and target (y)
X_train = train.drop('enrollment_count', axis=1)
y_train = train['enrollment_count']
X_test = test.drop('enrollment_count', axis=1)
y_test = test['enrollment_count']

## Split data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to try
models = [
    ('Linear Regression', LinearRegression(), {}),
    ('Random Forest', RandomForestRegressor(), {'n_estimators': [50, 100, 150]}),
    ('Gradient Boosting', GradientBoostingRegressor(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]})
]

# Perform cross-validation and hyperparameter tuning
results = []
for name, model, param_grid in models:
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    results.append((name, best_params, best_score))

# Write results to result.txt
with open('results.txt', 'w') as f:
    for name, best_params, best_score in results:
        f.write(f'{name}:\n')
        f.write(f'Best Parameters: {best_params}\n')
        f.write(f'Best Score: {best_score}\n')
        f.write('\n')

# Select the best model
best_model_name = min(results, key=lambda x: x[2])[0]
best_model = next(model[1] for model in models if model[0] == best_model_name)

# Train the final model for production
final_model = best_model.set_params(**results[0][1])  # Use best parameters from the selected model
X = pd.concat([X_train, X_test], axis = 0)
y = pd.concat([y_train, y_test], axis = 0)
final_model.fit(X, y)

# Save the final model with the current date in the filename
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
filename = f'final_model_{current_date}.joblib'
joblib.dump(final_model, filename)
print(f"Final model saved as {filename}")
