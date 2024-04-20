import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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

# K-Nearest Neighbors
knn_classifier = KNeighborsClassifier()

# Creating a pipeline
pipeline_knn = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', knn_classifier)])

# Fitting the model
pipeline_knn.fit(X_train, y_train)

# Evaluating the model on the validation set
knn_val_score = pipeline_knn.score(X_val, y_val)

print("Validation Score for KNN:", knn_val_score)

# Predictions using the KNN model
knn_predictions = pipeline_knn.predict(X_test)

# Creating a DataFrame to display actual and predicted values
knn_results = pd.DataFrame({'Actual Output': y_test, 'Predicted Output': knn_predictions})

# Save the results to a CSV file
knn_results.to_csv('knn_predictions.csv', index=False)

print(knn_results.head())

# Count of each class in actual and predicted outputs
class_counts_actual = knn_results['Actual Output'].value_counts()
class_counts_predicted = knn_results['Predicted Output'].value_counts()

# Creating a DataFrame for the bar plot
bar_plot_data = pd.DataFrame({'Actual': class_counts_actual, 'Predicted': class_counts_predicted})

# Creating the bar plot
plt.figure(figsize=(12, 6))
bar_plot_data.plot(kind='bar')
plt.title('Comparison of Actual and Predicted Class Counts (KNN)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Creating a histogram for predicted outputs
plt.figure(figsize=(10, 6))
plt.hist(knn_results['Predicted Output'], bins=len(knn_results['Predicted Output'].unique()), edgecolor='black')
plt.title('Histogram of Predicted Outputs (KNN)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
