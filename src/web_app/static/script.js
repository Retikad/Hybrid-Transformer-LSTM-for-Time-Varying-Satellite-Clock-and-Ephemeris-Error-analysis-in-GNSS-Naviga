// GNSS Error Prediction Dashboard - JavaScript

let currentFile = null;
let trainingInProgress = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupDragDrop();
    setupFileInput();
    updateButtonStates();
});

// File Upload Handling
function setupDragDrop() {
    const uploadArea = document.getElementById('uploadArea');

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('active');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('active');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('active');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', () => {
        document.getElementById('fileInput').click();
    });
}

function setupFileInput() {
    document.getElementById('fileInput').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

async function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            currentFile = data.data;
            displayFileInfo(data.data);
            updateButtonStates();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Upload failed: ' + error.message);
    }
}

function displayFileInfo(fileInfo) {
    document.getElementById('uploadArea').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'block';
    document.getElementById('fileName').textContent = fileInfo.filename;
    document.getElementById('fileRows').textContent = fileInfo.rows.toLocaleString();
    document.getElementById('fileCols').textContent = fileInfo.columns;
}

function changeFile() {
    currentFile = null;
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('fileInput').value = '';
    document.getElementById('trainingResults').style.display = 'none';
    updateButtonStates();
}

// Training
async function startTraining() {
    if (!currentFile || trainingInProgress) {
        alert('Please upload a file first');
        return;
    }

    trainingInProgress = true;
    
    const satelliteId = document.getElementById('satelliteId').value;
    const modelType = document.getElementById('modelType').value;
    const epochs = parseInt(document.getElementById('epochs').value);

    if (!satelliteId || epochs < 1) {
        alert('Please fill in all fields correctly');
        trainingInProgress = false;
        return;
    }

    // Show progress
    document.getElementById('trainingProgress').style.display = 'block';
    document.getElementById('trainingResults').style.display = 'none';
    document.getElementById('trainBtn').disabled = true;

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: currentFile.path,
                satellite_id: satelliteId,
                model_type: modelType,
                epochs: epochs
            })
        });

        const data = await response.json();

        if (response.ok && data.success) {
            displayTrainingResults(data.results);
        } else {
            alert('Training failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Training error: ' + error.message);
    } finally {
        trainingInProgress = false;
        document.getElementById('trainingProgress').style.display = 'none';
        document.getElementById('trainBtn').disabled = false;
        updateButtonStates();
    }

    // Poll for status
    const statusInterval = setInterval(async () => {
        try {
            const statusResponse = await fetch('/api/status');
            const statusData = await statusResponse.json();

            if (statusData.is_training) {
                updateProgress(statusData.progress, statusData.message);
            } else {
                clearInterval(statusInterval);
            }
        } catch (error) {
            clearInterval(statusInterval);
        }
    }, 1000);
}

function updateProgress(progress, message) {
    document.getElementById('progressFill').style.width = progress + '%';
    document.getElementById('progressFill').textContent = progress + '%';
    document.getElementById('progressText').textContent = message;
}

function displayTrainingResults(results) {
    document.getElementById('trainingProgress').style.display = 'none';
    document.getElementById('trainingResults').style.display = 'block';

    // Format numbers
    document.getElementById('metricRMSE').textContent = results.metrics.rmse.toFixed(4);
    document.getElementById('metricMAE').textContent = results.metrics.mae.toFixed(4);
    document.getElementById('metricR2').textContent = results.metrics.r2.toFixed(4);
    
    document.getElementById('trainSamples').textContent = results.train_samples.toLocaleString();
    document.getElementById('testSamples').textContent = results.test_samples.toLocaleString();
    document.getElementById('resultModelType').textContent = results.model_type.toUpperCase();

    updateButtonStates();
}

// Predictions
async function makePredictions() {
    if (!currentFile) {
        alert('Please upload a file first');
        return;
    }

    const modelType = document.getElementById('modelType').value;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: currentFile.path,
                model_type: modelType
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayPredictions(data.data);
        } else {
            alert('Prediction failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Prediction error: ' + error.message);
    }
}

function displayPredictions(predictionData) {
    document.getElementById('predictionResults').style.display = 'block';
    
    // Simple chart visualization
    const canvas = document.getElementById('predictionChart');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const predictions = predictionData.predictions;
    const actual = predictionData.actual || [];

    if (predictions.length === 0) {
        ctx.fillStyle = '#6b7280';
        ctx.font = 'bold 16px Arial';
        ctx.fillText('No data to display', canvas.width / 2 - 80, canvas.height / 2);
        return;
    }

    // Simple bar chart
    const chartWidth = canvas.width;
    const chartHeight = canvas.height;
    const maxValue = Math.max(...predictions.map(Math.abs));
    const barWidth = chartWidth / predictions.length;

    // Draw predictions
    ctx.fillStyle = '#2563eb';
    predictions.forEach((pred, i) => {
        const x = i * barWidth;
        const barHeight = (pred / maxValue) * (chartHeight - 40);
        ctx.fillRect(x, chartHeight - barHeight - 20, barWidth - 2, barHeight);
    });

    // Draw axis
    ctx.strokeStyle = '#d1d5db';
    ctx.beginPath();
    ctx.moveTo(0, chartHeight - 20);
    ctx.lineTo(chartWidth, chartHeight - 20);
    ctx.stroke();

    // Draw labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px Arial';
    ctx.fillText('Sample Index', chartWidth - 100, chartHeight);
}

// Download Demo Data
async function downloadDemoData() {
    try {
        const response = await fetch('/api/demo-data');
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'demo_gnss_data.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            alert('Failed to download demo data');
        }
    } catch (error) {
        alert('Download error: ' + error.message);
    }
}

// Update button states based on current state
function updateButtonStates() {
    const trainBtn = document.getElementById('trainBtn');
    const predictBtn = document.getElementById('predictBtn');

    trainBtn.disabled = !currentFile || trainingInProgress;
    
    const hasResults = document.getElementById('trainingResults').style.display === 'block';
    predictBtn.disabled = !currentFile || !hasResults;
}

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href !== '#') {
            e.preventDefault();
            const element = document.querySelector(href);
            if (element) {
                element.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }
    });
});
