import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score

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
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Function to calculate and print sensitivity for each model
def evaluate_model(model, X_val, y_val):
    # Creating a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', model)])

    # Fitting the model
    pipeline.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = pipeline.predict(X_val)

    # Calculate sensitivity for multiclass classification
    sensitivity = recall_score(y_val, y_pred, average='weighted', zero_division=1)  # Add zero_division parameter to handle zero division

    # Print sensitivity
    print(f"Sensitivity for {type(model).__name__}: {sensitivity:.4f}")

# Evaluate sensitivity for each model on the validation set
for name, model in models.items():
    evaluate_model(model, X_val, y_val)
