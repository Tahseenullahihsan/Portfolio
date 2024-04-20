
    function loadData() {
        fetch('http://http://127.0.0.1:5000/load_data')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                displayData(data);
            })
            .catch(error => {
                console.error('Error:', error.message);
                document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
            });
    }
    // display function
    function displayData(data) {
        const jsonContainer = document.getElementById('result-container');
        jsonContainer.innerHTML = '<h1>Load Data:</h1>';
        jsonContainer.innerHTML += '<p>' + data.message + '</p>';
        jsonContainer.innerHTML += '<pre>' + JSON.stringify(data.first_rows, null, 2) + '</pre>';
    }
    function prepareData() {
        fetch('http://127.0.0.1:5000/prepare_data')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
        const jsonContainer = document.getElementById('result-container');
        jsonContainer.innerHTML = '<h2>JSON Data:</h2>';
        jsonContainer.innerHTML += '<p>' + data.message + '</p>';
        jsonContainer.innerHTML += '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            })
            .catch(error => {
                console.error('Error:', error.message);
                document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
            });
    }

// c. Apply Logistic Regression and save weight files locally
function applyLogisticRegression() {
    fetch('http://127.0.0.1:5000/apply_logistic_regression')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayAlgorithmData(data);
        })
        .catch(error => {
            console.error('Error:', error.message);
            document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
        });
}

function applyDecisionTree() {
    fetch('http://127.0.0.1:5000/apply_decision_tree')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayAlgorithmData(data);
        })
        .catch(error => {
            console.error('Error:', error.message);
            document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
        });
}

function applyRandomForest() {
    fetch('http://127.0.0.1:5000/apply_random_forest')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayAlgorithmData(data);
        })
        .catch(error => {
            console.error('Error:', error.message);
            document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
        });
}


function applySVM() {
    fetch('http://127.0.0.1:5000/apply_svm')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayAlgorithmData(data);
        })
        .catch(error => {
            console.error('Error:', error.message);
            document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
        });
}

function applyKNN() {
    fetch('http://127.0.0.1:5000/apply_knn')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayAlgorithmData(data);
        })
        .catch(error => {
            console.error('Error:', error.message);
            document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
        });
}

function calculateConfusionMatrix() {
    fetch('http://127.0.0.1:5000/calculate_confusion_matrices')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const jsonContainer = document.getElementById('result-container');
            jsonContainer.innerHTML = '<h2>JSON Data:</h2>';
            jsonContainer.innerHTML += '<p>' + data.message + '</p>';
            jsonContainer.innerHTML += '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        })
        .catch(error => {
            console.error('Error:', error.message);
            document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
        });
}

function CalculateSensitivitySpecificity() {
    fetch('http://127.0.0.1:5000/CalculateSensitivitySpecificity')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const jsonContainer = document.getElementById('result-container');
            jsonContainer.innerHTML = '<h2>JSON Data:</h2>';
            jsonContainer.innerHTML += '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        })
        .catch(error => {
            console.error('Error:', error.message);
            document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
        });
}

function displayAlgorithmData(data) {
    const jsonContainer = document.getElementById('result-container');
    jsonContainer.innerHTML = '<h2>JSON Data:</h2>';

    // Display the message
    jsonContainer.innerHTML += '<p>' + data.message + '</p>';
    // Check if model_download_link exists in the data
    if (data.model_download_link) {
        // Create a button with a link to download the model
        const downloadButton = document.createElement('a');
        downloadButton.href = data.model_download_link;
        downloadButton.textContent = 'Download Model';
        downloadButton.classList.add('action-button'); // You can style it like your other buttons
        jsonContainer.appendChild(downloadButton);
    }
    jsonContainer.innerHTML += '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
}


function CalculateROCAUC() {
    fetch('http://127.0.0.1:5000/CalculateROCAUC')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const jsonContainer = document.getElementById('result-container');
            jsonContainer.innerHTML += '<p>' + data.message + '</p>';
        })
        .catch(error => {
            console.error('Error:', error.message);
            document.getElementById('result-container').innerHTML = '<p>Error occurred during API call.</p>';
        });
}

    function calculateSensitivityAtSpecificity() {
        // Simulate calculation
        const resultContainer = document.getElementById('result-container');
        resultContainer.innerHTML = '<p>Calculating Sensitivity at Specificity 0.90...</p>';

        // Implement your logic for Sensitivity at Specificity 0.90 here
        // Replace the innerHTML with the actual result or update it dynamically
        setTimeout(() => {
            resultContainer.innerHTML = '<p>Sensitivity at Specificity 0.90 calculated!</p>';
        }, 2000); // Simulating a delay, replace with actual logic
    }
