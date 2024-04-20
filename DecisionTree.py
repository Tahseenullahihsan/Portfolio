import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the car data set
data = pd.read_csv('car_eval_dataset.csv')

# Drop unnecessary index column if present,th coulmn contain the numric data
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# Splitting the dataset into features and target
X = data.drop('class', axis=1)
y = data['class']

# Splitting data into 70% train, 10% validation, and 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42)

# One-hot encoding for categorical variables, for the catagorical value
categorical_features = X_train.columns
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(transformers=[('cat', one_hot_encoder, categorical_features)])

# Decision Tree model apply
dt_classifier = DecisionTreeClassifier(random_state=42)

# Creating a pipeline
pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),('classifier', dt_classifier)])

# Parameters for GridSearchCV
param_grid_dt = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# GridSearchCV for hyperparameter tuning
grid_search_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=10, n_jobs=-1, verbose=1)

# Fitting the model. means traingin the model
grid_search_dt.fit(X_train, y_train)

# Best parameters and best score
best_params_dt = grid_search_dt.best_params_
best_score_dt = grid_search_dt.best_score_

print("Best Parameters:", best_params_dt)
print("Best Score:", best_score_dt)

# Predictions using the Decision Tree model,test the model
dt_predictions = grid_search_dt.predict(X_test)

# Creating a DataFrame to display actual and predicted values
dt_results = pd.DataFrame({'Actual Output': y_test, 'Predicted Output': dt_predictions})

# Save the results to a CSV file,decision_tree_prediction.csv
dt_results.to_csv('decision_tree_predictions.csv', index=False)

print(dt_results.head())

# Count of each class in actual and predicted outputs
class_counts_actual = dt_results['Actual Output'].value_counts()
class_counts_predicted = dt_results['Predicted Output'].value_counts()

# Creating a DataFrame for the bar plot
bar_plot_data = pd.DataFrame({'Actual': class_counts_actual, 'Predicted': class_counts_predicted})

# Creating the bar plot
plt.figure(figsize=(12, 6))
bar_plot_data.plot(kind='bar')
plt.title('Comparison of Actual and Predicted Class Counts')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Creating a histogram for predicted outputs
plt.figure(figsize=(10, 6))
plt.hist(dt_results['Predicted Output'], bins=len(dt_results['Predicted Output'].unique()), edgecolor='black')
plt.title('Histogram of Predicted Outputs')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
