import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('car_eval_dataset.csv')

# Drop unnecessary index column if present
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# Splitting the dataset into features and target
X = data.drop('class', axis=1)
y = data['class']

# Splitting data into 70% train, 10% validation, and 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42)

# One-hot encoding for categorical variables
categorical_features = X_train.columns
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(transformers=[('cat', one_hot_encoder, categorical_features)])

# Logistic Regression with L2 Regularization
lr_classifier = LogisticRegression(penalty='l2', random_state=42)

# Creating a pipeline
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', lr_classifier)])

# Fitting the model
pipeline_lr.fit(X_train, y_train)

# Evaluating the model on the validation set
lr_val_score = pipeline_lr.score(X_val, y_val)

print("Validation Score for Logistic Regression:", lr_val_score)

# Predictions using the Logistic Regression model
lr_predictions = pipeline_lr.predict(X_test)

# Creating a DataFrame to display actual and predicted values
lr_results = pd.DataFrame({'Actual Output': y_test, 'Predicted Output': lr_predictions})

# Save the results to a CSV file
lr_results.to_csv('logistic_regression_predictions.csv', index=False)

print(lr_results.head())

# Count of each class in actual and predicted outputs
class_counts_actual = lr_results['Actual Output'].value_counts()
class_counts_predicted = lr_results['Predicted Output'].value_counts()

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
plt.hist(lr_results['Predicted Output'], bins=len(lr_results['Predicted Output'].unique()), edgecolor='black')
plt.title('Histogram of Predicted Outputs')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
