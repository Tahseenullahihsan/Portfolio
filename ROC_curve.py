import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

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

# Models
models = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Plot ROC curve for each model
plt.figure(figsize=(10, 8))

for name, model in models.items():
    # Creating a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', model)])

    # Fitting the model
    pipeline.fit(X_train, y_train)

    # Predict probabilities for the positive class
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    # Convert the problem into binary classification
    y_binary = (y_val == 'good').astype(int)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_binary, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plotting ROC curve
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plotting the ROC curve for a random classifier (baseline)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random', alpha=0.5)

# Customize the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()
