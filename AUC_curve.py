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
from sklearn.metrics import roc_auc_score
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

# Classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),  # Ensure probability is set to True for SVC
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Initialize variables to keep track of the best classifier and its AUC score
best_classifier = None
max_auc_score = 0.0

# Loop through classifiers
for name, classifier in classifiers.items():
    # Creating a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', classifier)])

    # Fitting the model
    pipeline.fit(X_train, y_train)

    # Evaluating the model on the validation set
    if len(set(y_val)) == 2:  # Binary classification
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_val_proba)
    else:  # Multiclass classification
        y_val_proba = pipeline.predict_proba(X_val)
        auc_score = roc_auc_score(pd.get_dummies(y_val), y_val_proba, multi_class='ovr')

    print(f"AUC for {name}: {auc_score}")

    # Update the best classifier if the current one has a higher AUC score
    if auc_score > max_auc_score:
        max_auc_score = auc_score
        best_classifier = pipeline

# Now, you can use `best_classifier` for predictions on the test set
if len(set(y_test)) == 2:  # Binary classification
    y_test_proba = best_classifier.predict_proba(X_test)[:, 1]
    test_auc_score = roc_auc_score(y_test, y_test_proba)
else:  # Multiclass classification
    y_test_proba = best_classifier.predict_proba(X_test)
    test_auc_score = roc_auc_score(pd.get_dummies(y_test), y_test_proba, multi_class='ovr')

print(f"AUC on the Test Set: {test_auc_score}")
