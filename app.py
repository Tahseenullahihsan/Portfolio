from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict

app = Flask(__name__)
CORS(app)


# API endpoint to load data and display first rows
@app.route('/load_data', methods=['GET'])
def load_data():
    global dataset

    try:
        # Replace 'car_eval_dataset.csv' with the actual name of your dataset CSV file
        dataset = pd.read_csv('car_eval_dataset.csv')

        # Display first few rows of the dataset
        first_rows = dataset.head().to_dict(orient='records')

        return jsonify({"message": "Data loaded successfully! First five rows of Data are Printed below",
                        "first_rows": first_rows})
    except Exception as e:
        return jsonify({"error": f"Error loading data: {str(e)}"}), 500


# Continue with other API endpoints (e.g., /prepare_data, /apply_random_forest, /calculate_metrics, etc.)

# API endpoint to prepare data for training

@app.route('/prepare_data', methods=['GET', 'POST'])
def prepare_data():
    try:
        # Assuming 'class' is the target variable
        X = dataset.drop("class", axis=1)
        y = dataset["class"]
        # Encode categorical features using one-hot encoding
        categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        cat_data = X[categorical_columns]

        one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_data = one_hot_encoder.fit_transform(cat_data)
        encoded_column_names = one_hot_encoder.get_feature_names_out(categorical_columns)
        X_encoded = pd.DataFrame(encoded_data, columns=encoded_column_names)

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42)

        response = {
            "message": "Data preparation successful",
            "X_train": X_train.head().to_dict(orient='records'),
            "X_val": X_val.head().to_dict(orient='records'),
            "X_test": X_test.head().to_dict(orient='records'),
            "y_train": y_train.head().tolist(),
            "y_val": y_val.tolist(),
            "y_test": y_test.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


# Placeholder variables for dataset and models
dataset = None
logistic_model = None  # Assuming you have a global LogisticRegression variable


# API endpoint for Logistic Regression
@app.route('/apply_logistic_regression', methods=['GET', 'POST'])
def apply_logistic_regression():
    global dataset, logistic_model

    try:
        # Check if the dataset is loaded
        if dataset is None:
            return jsonify({"error": "Dataset not loaded. Use /load_data endpoint first."}), 400

        # Assuming 'class' is the target variable
        X = dataset.drop("class", axis=1)
        y = dataset["class"]

        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Create and train Logistic Regression model
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(X_train, y_train)

        # Save the trained model using pickle
        with open('logistic_regression_model.pkl', 'wb') as model_file:
            pickle.dump(logistic_model, model_file)
        logistic_pred = logistic_model.predict(X_encoded)
        # Calculate accuracy
        accuracy = accuracy_score(dataset['class'], logistic_pred)

        # Generate classification report
        class_report_dict = classification_report(dataset['class'], logistic_pred, target_names=logistic_model.classes_,
                                                  output_dict=True)

        # Sort the dictionary by class labels
        sorted_class_report = {k: class_report_dict[k] for k in sorted(class_report_dict)}
        # Provide a link to download the model file
        model_download_link = f"{request.url_root}download_model"
        return jsonify({
            "message": "Logistic Regression applied and weights saved",
            "model_download_link": model_download_link,
            "svm_predictions": logistic_pred.tolist(),
            "accuracy": accuracy,
            "classification_report": sorted_class_report
        })

    except Exception as e:
        return jsonify({"error": f"Error applying Logistic Regression: {str(e)}"}), 500


# API endpoint to download the logistic regression model
@app.route('/download_model', methods=['GET'])
def download_model():
    try:
        # Check if the model file exists
        if os.path.exists('logistic_regression_model.pkl'):
            return send_file('logistic_regression_model.pkl', as_attachment=True)
        else:
            return jsonify({"error": "Model file not found."})

    except Exception as e:
        return jsonify({"error": str(e)})


decision_tree_model = None  # Assuming you have a global DecisionTreeClassifier variable


# API endpoint for Decision Tree
@app.route('/apply_decision_tree', methods=['GET', 'POST'])
def apply_decision_tree():
    global dataset, decision_tree_model

    try:
        # Check if the dataset is loaded
        if dataset is None:
            return jsonify({"error": "Dataset not loaded. Use /load_data endpoint first."}), 400

        # Assuming 'class' is the target variable
        X = dataset.drop("class", axis=1)
        y = dataset["class"]

        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Create and train Decision Tree model
        decision_tree_model = DecisionTreeClassifier()
        decision_tree_model.fit(X_train, y_train)

        # Save the trained model using pickle
        with open('decision_tree_model.pkl', 'wb') as model_file:
            pickle.dump(decision_tree_model, model_file)

        decision_tree_pred = decision_tree_model.predict(X_encoded)

        # Calculate accuracy
        accuracy = accuracy_score(dataset['class'], decision_tree_pred)

        # Generate classification report
        class_report_dict = classification_report(dataset['class'], decision_tree_pred,
                                                  target_names=decision_tree_model.classes_, output_dict=True)

        # Sort the dictionary by class labels
        sorted_class_report = {k: class_report_dict[k] for k in sorted(class_report_dict)}

        # Provide a link to download the model file
        model_download_link = f"{request.url_root}download_model"
        return jsonify({
            "message": "Decision Tree applied and weights saved",
            "model_download_link": model_download_link,
            "decision_tree_predictions": decision_tree_pred.tolist(),
            "accuracy": accuracy,
            "classification_report": sorted_class_report
        })

    except Exception as e:
        return jsonify({"error": f"Error applying Decision Tree: {str(e)}"}), 500


# API endpoint to download the Decision Tree model
@app.route('/download_decision_tree_model', methods=['GET'])
def ddownload_decision_tree_model():
    try:
        # Check if the model file exists
        if os.path.exists('decision_tree_model.pkl'):
            return send_file('decision_tree_model.pkl', as_attachment=True)
        else:
            return jsonify({"error": "Model file not found."})

    except Exception as e:
        return jsonify({"error": str(e)})


random_forest_model = None  # Assuming you have a global RandomForestClassifier variable


# API endpoint for Random Forest
@app.route('/apply_random_forest', methods=['GET', 'POST'])
def apply_random_forest():
    global dataset, random_forest_model

    try:
        # Check if the dataset is loaded
        if dataset is None:
            return jsonify({"error": "Dataset not loaded. Use /load_data endpoint first."}), 400

        # Assuming 'class' is the target variable
        X = dataset.drop("class", axis=1)
        y = dataset["class"]

        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Create and train Random Forest model
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)

        # Save the trained model using pickle
        with open('random_forest_model.pkl', 'wb') as model_file:
            pickle.dump(random_forest_model, model_file)

        random_forest_pred = random_forest_model.predict(X_encoded)

        # Calculate accuracy
        accuracy = accuracy_score(dataset['class'], random_forest_pred)

        # Generate classification report
        class_report_dict = classification_report(dataset['class'], random_forest_pred,
                                                  target_names=random_forest_model.classes_, output_dict=True)

        # Sort the dictionary by class labels
        sorted_class_report = {k: class_report_dict[k] for k in sorted(class_report_dict)}

        # Provide a link to download the model file
        model_download_link = f"{request.url_root}download_random_forest_model"
        return jsonify({
            "message": "Random Forest applied and weights saved",
            "model_download_link": model_download_link,
            "random_forest_predictions": random_forest_pred.tolist(),
            "accuracy": accuracy,
            "classification_report": sorted_class_report
        })

    except Exception as e:
        return jsonify({"error": f"Error applying Random Forest: {str(e)}"}), 500


# API endpoint to download the Random Forest model
@app.route('/download_random_forest_model', methods=['GET'])
def download_random_forest_model():
    try:
        # Check if the model file exists
        if os.path.exists('random_forest_model.pkl'):
            return send_file('random_forest_model.pkl', as_attachment=True)
        else:
            return jsonify({"error": "Model file not found."})

    except Exception as e:
        return jsonify({"error": str(e)})


svm_model = None  # Assuming you have a global Support Vector Machine (SVM) variable


# API endpoint for Support Vector Machine (SVM)
@app.route('/apply_svm', methods=['GET', 'POST'])
def apply_svm():
    global dataset, svm_model

    try:
        # Check if the dataset is loaded
        if dataset is None:
            return jsonify({"error": "Dataset not loaded. Use /load_data endpoint first."}), 400

        # Assuming 'class' is the target variable
        X = dataset.drop("class", axis=1)
        y = dataset["class"]

        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Create and train Support Vector Machine (SVM) model
        svm_model = SVC()
        svm_model.fit(X_train, y_train)

        # Save the trained model using pickle
        with open('svm_model.pkl', 'wb') as model_file:
            pickle.dump(svm_model, model_file)

        svm_pred = svm_model.predict(X_encoded)

        # Calculate accuracy
        accuracy = accuracy_score(dataset['class'], svm_pred)

        # Generate classification report
        class_report_dict = classification_report(dataset['class'], svm_pred, target_names=svm_model.classes_,
                                                  output_dict=True)

        # Sort the dictionary by class labels
        sorted_class_report = {k: class_report_dict[k] for k in sorted(class_report_dict)}

        # Provide a link to download the model file
        model_download_link = f"{request.url_root}download_svm_model"
        return jsonify({
            "message": "Support Vector Machine (SVM) applied and weights saved",
            "model_download_link": model_download_link,
            "svm_predictions": svm_pred.tolist(),
            "accuracy": accuracy,
            "classification_report": sorted_class_report
        })

    except Exception as e:
        return jsonify({"error": f"Error applying Support Vector Machine (SVM): {str(e)}"}), 500


# API endpoint to download the SVM model
@app.route('/download_svm_model', methods=['GET'])
def download_svm_model():
    try:
        # Check if the model file exists
        if os.path.exists('svm_model.pkl'):
            return send_file('svm_model.pkl', as_attachment=True)
        else:
            return jsonify({"error": "Model file not found."})

    except Exception as e:
        return jsonify({"error": str(e)})


dataset = None
knn_model = None


@app.route('/apply_knn', methods=['GET', 'POST'])
def apply_knn():
    global dataset, knn_model

    try:
        if dataset is None:
            return jsonify({"error": "Dataset not loaded. Use /load_data endpoint first."}), 400

        X = dataset.drop("class", axis=1)
        y = dataset["class"]

        X_encoded = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)

        with open('knn_model.pkl', 'wb') as model_file:
            pickle.dump(knn_model, model_file)

        knn_pred = knn_model.predict(X_encoded)

        accuracy = accuracy_score(dataset['class'], knn_pred)

        class_report_dict = classification_report(dataset['class'], knn_pred, target_names=knn_model.classes_,
                                                  output_dict=True)

        sorted_class_report = {k: class_report_dict[k] for k in sorted(class_report_dict)}

        model_download_link = f"{request.url_root}download_knn_model"
        return jsonify({
            "message": "K-Nearest Neighbors (KNN) applied and weights saved",
            "model_download_link": model_download_link,
            "knn_predictions": knn_pred.tolist(),
            "accuracy": accuracy,
            "classification_report": sorted_class_report
        })

    except Exception as e:
        return jsonify({"error": f"Error applying K-Nearest Neighbors (KNN): {str(e)}"}), 500


@app.route('/download_knn_model', methods=['GET'])
def download_knn_model():
    try:
        if os.path.exists('knn_model.pkl'):
            return send_file('knn_model.pkl', as_attachment=True)
        else:
            return jsonify({"error": "Model file not found."})

    except Exception as e:
        return jsonify({"error": str(e)})


# Load the dataset
dataset = pd.read_csv("car_eval_dataset.csv")

# Encode categorical features using OneHotEncoder
categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
cat_data = dataset[categorical_columns]

one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_data = one_hot_encoder.fit_transform(cat_data)
encoded_column_names = one_hot_encoder.get_feature_names_out(categorical_columns)
X_encoded = pd.DataFrame(encoded_data, columns=encoded_column_names)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, dataset['class'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
y_pred_logistic = cross_val_predict(logistic_model, X_train, y_train, cv=10)
logistic_model.fit(X_train, y_train)
y_test_pred_logistic = logistic_model.predict(X_test)
conf_matrix_logistic = confusion_matrix(y_test, y_test_pred_logistic)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
y_pred_dt = cross_val_predict(dt_model, X_train, y_train, cv=10)
dt_model.fit(X_train, y_train)
y_test_pred_dt = dt_model.predict(X_test)
conf_matrix_dt = confusion_matrix(y_test, y_test_pred_dt)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
y_pred_rf = cross_val_predict(rf_model, X_train, y_train, cv=10)
rf_model.fit(X_train, y_train)
y_test_pred_rf = rf_model.predict(X_test)
conf_matrix_rf = confusion_matrix(y_test, y_test_pred_rf)

# SVM
svm_model = SVC(random_state=42)
y_pred_svm = cross_val_predict(svm_model, X_train, y_train, cv=10)
svm_model.fit(X_train, y_train)
y_test_pred_svm = svm_model.predict(X_test)
conf_matrix_svm = confusion_matrix(y_test, y_test_pred_svm)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
y_pred_knn = cross_val_predict(knn_model, X_train, y_train, cv=10)
knn_model.fit(X_train, y_train)
y_test_pred_knn = knn_model.predict(X_test)
conf_matrix_knn = confusion_matrix(y_test, y_test_pred_knn)


# Define endpoint to get confusion matrices
@app.route('/calculate_confusion_matrices', methods=['GET'])
def calculate_confusion_matrices():
    try:
        return jsonify({
            "message": "Confusion Matrix Calculated for each algorithm",
            'conf_matrix_logistic': conf_matrix_logistic.tolist(),
            'conf_matrix_dt': conf_matrix_dt.tolist(),
            'conf_matrix_rf': conf_matrix_rf.tolist(),
            'conf_matrix_svm': conf_matrix_svm.tolist(),
            'conf_matrix_knn': conf_matrix_knn.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


dataset = pd.read_csv("car_eval_dataset.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
cat_data = X[categorical_columns]

one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_data = one_hot_encoder.fit_transform(cat_data)
encoded_column_names = one_hot_encoder.get_feature_names_out(categorical_columns)
X_encoded = pd.DataFrame(encoded_data, columns=encoded_column_names)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

for model_name, model in models.items():
    y_pred = cross_val_predict(model, X_train, y_train, cv=10)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    sensitivity = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    specificity = 1 - (conf_matrix.sum(axis=0) - conf_matrix.diagonal()) / conf_matrix.sum(axis=0)
    print(f"\nResults for {model_name}:")
    for i in range(len(model.classes_)):
        print(f"Class {model.classes_[i]} - Sensitivity: {sensitivity[i]:.2f}, Specificity: {specificity[i]:.2f}")


@app.route('/CalculateSensitivitySpecificity', methods=['GET', 'POST'])
def calculate_sensitivity_specificity_all_models():
    try:
        result_dict = {}

        for model_name, model in models.items():
            y_test_pred = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            sensitivity = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
            specificity = 1 - (conf_matrix.sum(axis=0) - conf_matrix.diagonal()) / conf_matrix.sum(axis=0)

            class_results = {}
            for i in range(len(model.classes_)):
                class_results[f'Class {model.classes_[i]}'] = {
                    'Sensitivity': float(sensitivity[i]),
                    'Specificity': float(specificity[i])
                }

            result_dict[model_name] = class_results

        return jsonify(result_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


dataset = pd.read_csv("car_eval_dataset.csv")

# Split the dataset into features (X) and target variable (y) using iloc
X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]  # Target variable

# Encode categorical features using OneHotEncoder
categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
cat_data = X[categorical_columns]

one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_data = one_hot_encoder.fit_transform(cat_data)
encoded_column_names = one_hot_encoder.get_feature_names_out(categorical_columns)
X_encoded = pd.DataFrame(encoded_data, columns=encoded_column_names)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Define models
models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}


# Define endpoint to calculate and plot ROC curve for all models
@app.route('/CalculateROCAUC', methods=['GET', 'POST'])
def calculate_roc_auc_all_models():
    try:
        plt.figure(figsize=(8, 6))

        for model_name, model in models.items():
            # Cross-validation prediction on the training set
            y_pred = cross_val_predict(model, X_train, y_train, cv=10)

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Predictions on the test set
            y_test_pred = model.predict(X_test)

            for i in range(len(model.classes_)):
                y_binary = (y_test == model.classes_[i]).astype(int)
                y_prob = model.predict_proba(X_test)[:, i]
                fpr, tpr, _ = roc_curve(y_binary, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} - Class {model.classes_[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass Receiver Operating Characteristic (ROC) Curve for All Models')
        plt.legend(loc='lower right')
        plt.show()

        return jsonify({'message': 'ROC curves plotted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Load the dataset
dataset = pd.read_csv("car_eval_dataset.csv")

# Split the dataset into features (X) and target variable (y) using iloc
X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]  # Target variable

# Encode categorical features using OneHotEncoder
categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
cat_data = X[categorical_columns]

one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_data = one_hot_encoder.fit_transform(cat_data)
encoded_column_names = one_hot_encoder.get_feature_names_out(categorical_columns)
X_encoded = pd.DataFrame(encoded_data, columns=encoded_column_names)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Implement Logistic Regression with a maximum iteration of 1000
logistic_model = LogisticRegression(max_iter=1000)

# Cross-validation prediction on the training set
y_pred_logistic = cross_val_predict(logistic_model, X_train, y_train, cv=10)

# Fit the model on the training data
logistic_model.fit(X_train, y_train)

# Predictions on the test set
y_test_pred_logistic = logistic_model.predict(X_test)


# Define endpoint to calculate sensitivity at specificity 0.90
@app.route('/CalculateSensitivityAtSpecificity', methods=['GET', 'POST'])
def calculate_sensitivity_at_specificity():
    try:
        desired_specificity = 0.90
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_logistic)
        roc_auc = auc(fpr, tpr)

        # Find the index of the threshold that is closest to the desired specificity
        idx = np.argmax(fpr >= (1 - desired_specificity))

        # Get the corresponding threshold
        threshold_at_desired_specificity = round(thresholds[idx], 4)

        # Get the corresponding TPR (sensitivity)
        sensitivity_at_desired_specificity = round(tpr[idx], 4)

        return jsonify({
            'sensitivity_at_specificity_0.90': sensitivity_at_desired_specificity,
            'threshold_at_specificity_0.90': threshold_at_desired_specificity
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
