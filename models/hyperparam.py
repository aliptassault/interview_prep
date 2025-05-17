from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


iris_data= load_iris()
X=iris_data.data
y=iris_data.target
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# Define parameter distribution for random search
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'objective': ['multi:softmax'],
    'num_class': [3],
    'seed': [42]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(),
    param_distributions=param_dist,
    n_iter=25,
    scoring='accuracy',
    cv=5,
    random_state=42,
    verbose=1
)

# Fit the random search
random_search.fit(X_train, y_train)

# Get best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score: {:.2f}".format(random_search.best_score_))

# Predict using the best model
y_pred = random_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy with best parameters: {accuracy:.2f}")