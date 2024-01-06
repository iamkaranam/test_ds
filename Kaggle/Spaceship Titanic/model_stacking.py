from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression


# Assuming you have your train_df and test_df dataframes
# Split the training data into features (X) and target variable (y)

# Split the training data into training and validation sets
X_train_base, X_val, y_train_base, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models
base_models = [
    ('rf', RandomForestClassifier(random_state=42,n_estimators=200,max_depth=7 ) ),
    ('ada', AdaBoostClassifier(random_state=1,n_estimators=750,learning_rate=0.7) ),
    ('gb', GradientBoostingClassifier(learning_rate=0.008,random_state=1,n_estimators=5000) ),
    ('xgb', xgb.XGBClassifier(random_state=1,learning_rate=0.05,n_estimators = 500) )
]

# Define the stacking classifier with a meta-model (Logistic Regression is used here, but you can choose another model)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

# Train the stacking model on the base models
stacking_model.fit(X_train_base, y_train_base)

# Make predictions on the validation set
stacking_val_preds = stacking_model.predict(X_val)

# Evaluate the performance of the stacking model
accuracy = accuracy_score(y_val, stacking_val_preds)
print(f'Stacking Model Accuracy: {accuracy}')

# Now, you can train the stacking model on the entire training set and make predictions on the test set
stacking_model.fit(X_train, y_train)
test_preds = stacking_model.predict(test_df)

# 'test_preds' now contains the final predictions for the test set


import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score




# Define base models
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=7)
ada_model = AdaBoostClassifier(random_state=1, n_estimators=750, learning_rate=0.7)
gb_model = GradientBoostingClassifier(learning_rate=0.008, random_state=1, n_estimators=5000)
xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.05, n_estimators=500)

# Create a stacking classifier
stacking_model = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('ada', ada_model),
        ('gb', gb_model),
        ('xgb', xgb_model)
    ],
    final_estimator=xgb_model,  # You can choose any model as the final estimator
    cv=StratifiedKFold(n_splits=10),  # Use StratifiedKFold for classification tasks
    stack_method='predict_proba',  # Use 'predict_proba' for classifiers
    n_jobs=-1  # Use all available CPU cores
)

# Make predictions using cross-validation
stacking_predictions = cross_val_predict(stacking_model, X, y, cv=StratifiedKFold(n_splits=10))

# Evaluate the performance
accuracy = accuracy_score(y, stacking_predictions)
print(f'Stacking Classifier Accuracy: {accuracy}')

# Train the stacking model on the entire dataset
stacking_model.fit(X, y)

# Now you can use the trained stacking model to make predictions on your test set (test_df)
test_df = test_df[list(X.columns)]

y_pred_test = stacking_model.predict(test_df)

test_passenger_ids['Transported'] = y_pred_test
test_passenger_ids['Transported'] = test_passenger_ids['Transported'].astype(bool)

test_passenger_ids.to_csv('./Documents/Vamsi/DS/Kaggle/Spaceship Titanic/submission_12.csv',index = False)
