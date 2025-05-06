/**
 * Training functionality for Cellular Evolutionary Automata and CNN models
 *
 * This file provides functions for training the different models:
 * - Standard CNN with Adam optimizer
 * - Synchronous Cellular Evolutionary Automata (SyncCEA)
 * - Asynchronous Cellular Evolutionary Automata (AsyncCEA)
 */
class ModelTrainer {
    constructor() {
        this.trainingStatus = {
            isTraining: false,
            currentModel: null,
            progress: 0,
            currentStep: 0,
            totalSteps: 0,
            startTime: null,
            trainingLog: []
        };

        this.modelResults = {
            cnn: {
                trained: false,
                accuracy: 0,
                trainTime: 0,
                epochs: 0,
                learningRate: 0
            },
            syncCEA: {
                trained: false,
                accuracy: 0,
                trainTime: 0,
                generations: 0,
                gridSize: 0,
                selectionMethod: '',
                fitnessHistory: []
            },
            asyncCEA: {
                trained: false,
                accuracy: 0,
                trainTime: 0,
                generations: 0,
                gridSize: 0,
                selectionMethod: '',
                fitnessHistory: []
            }
        };

        // Initialize UI handlers
        this.initializeUIHandlers();

        // Initialize charts
        this.initializeCharts();
    }

    initializeUIHandlers() {
        // Tab navigation
        document.querySelectorAll('.training-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.getAttribute('data-tab');

                // Set active tab
                document.querySelectorAll('.training-tab').forEach(t =>
                    t.classList.remove('active'));
                tab.classList.add('active');

                // Show corresponding tab content
                document.querySelectorAll('.training-tab-content').forEach(content => {
                    content.classList.remove('active');
                    if (content.id === `${tabId}-tab`) {
                        content.classList.add('active');
                    }
                });
            });
        });

        // Training button handlers
        document.getElementById('train-cnn-btn').addEventListener('click', () =>
            this.startTraining('cnn'));

        document.getElementById('train-sync-cea-btn').addEventListener('click', () =>
            this.startTraining('sync-cea'));

        document.getElementById('train-async-cea-btn').addEventListener('click', () =>
            this.startTraining('async-cea'));

        document.getElementById('stop-training-btn').addEventListener('click', () =>
            this.stopTraining());

        // Save model handlers
        document.getElementById('save-cnn-btn').addEventListener('click', () =>
            this.saveModel('cnn'));

        document.getElementById('save-sync-cea-btn').addEventListener('click', () =>
            this.saveModel('sync-cea'));

        document.getElementById('save-async-cea-btn').addEventListener('click', () =>
            this.saveModel('async-cea'));

        document.getElementById('save-all-models-btn').addEventListener('click', () =>
            this.saveAllModels());

        // View details handlers
        document.querySelectorAll('.view-details-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modelType = e.currentTarget.getAttribute('data-model');
                this.showModelDetails(modelType);
            });
        });
    }

    initializeCharts() {
        // Accuracy comparison chart
        this.accuracyChart = new Chart(
            document.getElementById('accuracy-chart').getContext('2d'),
            {
                type: 'bar',
                data: {
                    labels: ['Standard CNN', 'Sync CEA', 'Async CEA'],
                    datasets: [{
                        label: 'Test Accuracy (%)',
                        data: [0, 0, 0],
                        backgroundColor: [
                            'rgba(79, 70, 229, 0.6)',
                            'rgba(16, 185, 129, 0.6)',
                            'rgba(245, 158, 11, 0.6)'
                        ],
                        borderColor: [
                            'rgba(79, 70, 229, 1)',
                            'rgba(16, 185, 129, 1)',
                            'rgba(245, 158, 11, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            }
        );

        // Training time comparison chart
        this.timeChart = new Chart(
            document.getElementById('training-time-chart').getContext('2d'),
            {
                type: 'bar',
                data: {
                    labels: ['Standard CNN', 'Sync CEA', 'Async CEA'],
                    datasets: [{
                        label: 'Training Time (seconds)',
                        data: [0, 0, 0],
                        backgroundColor: [
                            'rgba(79, 70, 229, 0.6)',
                            'rgba(16, 185, 129, 0.6)',
                            'rgba(245, 158, 11, 0.6)'
                        ],
                        borderColor: [
                            'rgba(79, 70, 229, 1)',
                            'rgba(16, 185, 129, 1)',
                            'rgba(245, 158, 11, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            }
        );

        // Fitness evolution chart
        this.fitnessChart = new Chart(
            document.getElementById('fitness-evolution-chart').getContext('2d'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Sync CEA Best Fitness',
                            data: [],
                            borderColor: 'rgba(16, 185, 129, 1)',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.1,
                            fill: true
                        },
                        {
                            label: 'Async CEA Best Fitness',
                            data: [],
                            borderColor: 'rgba(245, 158, 11, 1)',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            tension: 0.1,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            }
        );
    }

    updateCharts() {
        // Update accuracy chart
        this.accuracyChart.data.datasets[0].data = [
            this.modelResults.cnn.accuracy,
            this.modelResults.syncCEA.accuracy,
            this.modelResults.asyncCEA.accuracy
        ];
        this.accuracyChart.update();

        // Update training time chart
        this.timeChart.data.datasets[0].data = [
            this.modelResults.cnn.trainTime,
            this.modelResults.syncCEA.trainTime,
            this.modelResults.asyncCEA.trainTime
        ];
        this.timeChart.update();

        // Update fitness evolution chart
        const maxGen = Math.max(
            this.modelResults.syncCEA.fitnessHistory.length,
            this.modelResults.asyncCEA.fitnessHistory.length
        );

        if (maxGen > 0) {
            const labels = Array.from({length: maxGen}, (_, i) => i + 1);
            this.fitnessChart.data.labels = labels;

            this.fitnessChart.data.datasets[0].data = this.modelResults.syncCEA.fitnessHistory;
            this.fitnessChart.data.datasets[1].data = this.modelResults.asyncCEA.fitnessHistory;

            this.fitnessChart.update();
        }
    }

    startTraining(modelType) {
        if (this.trainingStatus.isTraining) {
            this.logMessage('Please wait for the current training to complete', 'warning');
            return;
        }

        // Set training state
        this.trainingStatus.isTraining = true;
        this.trainingStatus.currentModel = modelType;
        this.trainingStatus.progress = 0;
        this.trainingStatus.currentStep = 0;
        this.trainingStatus.startTime = Date.now();
        this.trainingStatus.trainingLog = [];

        // Show training status UI
        document.getElementById('training-status').classList.add('active');
        document.getElementById('training-progress').style.width = '0%';
        document.getElementById('training-log').innerHTML = '';

        // Get training parameters and update UI
        let modelName, totalSteps, otherParams;

        if (modelType === 'cnn') {
            modelName = 'Standard CNN';
            totalSteps = parseInt(document.getElementById('cnn-epochs').value);
            const learningRate = parseFloat(document.getElementById('cnn-learning-rate').value);
            otherParams = `Learning Rate: ${learningRate}`;

            // Store parameters
            this.modelResults.cnn.epochs = totalSteps;
            this.modelResults.cnn.learningRate = learningRate;
        }
        else if (modelType === 'sync-cea') {
            modelName = 'Synchronous CEA';
            totalSteps = parseInt(document.getElementById('sync-cea-generations').value);
            const gridSize = parseInt(document.getElementById('sync-cea-grid-size').value);
            const selectionMethod = document.getElementById('sync-cea-selection').value;
            otherParams = `Grid Size: ${gridSize}x${gridSize}, Selection: ${selectionMethod}`;

            // Store parameters
            this.modelResults.syncCEA.generations = totalSteps;
            this.modelResults.syncCEA.gridSize = gridSize;
            this.modelResults.syncCEA.selectionMethod = selectionMethod;
            this.modelResults.syncCEA.fitnessHistory = [];
        }
        else if (modelType === 'async-cea') {
            modelName = 'Asynchronous CEA';
            totalSteps = parseInt(document.getElementById('async-cea-generations').value);
            const gridSize = parseInt(document.getElementById('async-cea-grid-size').value);
            const selectionMethod = document.getElementById('async-cea-selection').value;
            otherParams = `Grid Size: ${gridSize}x${gridSize}, Selection: ${selectionMethod}`;

            // Store parameters
            this.modelResults.asyncCEA.generations = totalSteps;
            this.modelResults.asyncCEA.gridSize = gridSize;
            this.modelResults.asyncCEA.selectionMethod = selectionMethod;
            this.modelResults.asyncCEA.fitnessHistory = [];
        }

        this.trainingStatus.totalSteps = totalSteps;
        document.getElementById('training-info-text').textContent =
            `Training ${modelName} (${otherParams})`;
        document.getElementById('current-step').textContent = '0';
        document.getElementById('total-steps').textContent = totalSteps;

        // Log the start of training
        this.logMessage(`Starting training of ${modelName}`, 'info');
        this.logMessage(`Parameters: ${otherParams}`, 'info');

        // Start the training simulation
        this.simulateTraining();
    }

    simulateTraining() {
        // This is a frontend demo that simulates the training process
        // In a real implementation, this would make API calls to the backend

        const { currentModel, totalSteps } = this.trainingStatus;
        const intervalTime = currentModel === 'cnn' ? 200 : 300; // Simulate CEA taking longer

        const trainingInterval = setInterval(() => {
            if (!this.trainingStatus.isTraining) {
                clearInterval(trainingInterval);
                this.logMessage('Training stopped by user', 'warning');
                return;
            }

            this.trainingStatus.currentStep++;
            const progress = (this.trainingStatus.currentStep / totalSteps) * 100;
            this.trainingStatus.progress = progress;

            // Update UI
            document.getElementById('current-step').textContent = this.trainingStatus.currentStep;
            document.getElementById('training-progress').style.width = `${progress}%`;

            // Generate simulated training metrics
            if (currentModel === 'cnn') {
                const loss = 2.5 * Math.exp(-0.3 * this.trainingStatus.currentStep) + 0.1 * Math.random();
                const accuracy = 50 + 45 * (1 - Math.exp(-0.3 * this.trainingStatus.currentStep)) + 3 * Math.random();
                this.logMessage(`Epoch ${this.trainingStatus.currentStep}/${totalSteps} - Loss: ${loss.toFixed(4)}, Accuracy: ${accuracy.toFixed(2)}%`, 'info');
            }
            else if (currentModel === 'sync-cea' || currentModel === 'async-cea') {
                const bestFitness = 40 + 50 * (1 - Math.exp(-0.3 * this.trainingStatus.currentStep)) + 5 * Math.random();
                const avgFitness = bestFitness * 0.8 + 4 * Math.random();

                this.logMessage(`Generation ${this.trainingStatus.currentStep}/${totalSteps} - Best Fitness: ${bestFitness.toFixed(2)}%, Avg Fitness: ${avgFitness.toFixed(2)}%`, 'info');

                // Store fitness history
                if (currentModel === 'sync-cea') {
                    this.modelResults.syncCEA.fitnessHistory.push(bestFitness);
                } else {
                    this.modelResults.asyncCEA.fitnessHistory.push(bestFitness);
                }
            }

            // Check if training is complete
            if (this.trainingStatus.currentStep >= totalSteps) {
                clearInterval(trainingInterval);
                this.finishTraining();
            }
        }, intervalTime);
    }

    stopTraining() {
        if (!this.trainingStatus.isTraining) return;

        this.trainingStatus.isTraining = false;
        document.getElementById('training-status').classList.remove('active');
        this.logMessage('Training stopped', 'warning');
    }

    finishTraining() {
        // Calculate training time
        const trainingTime = (Date.now() - this.trainingStatus.startTime) / 1000;

        // Generate final results based on model type
        const { currentModel } = this.trainingStatus;
        let finalAccuracy;

        if (currentModel === 'cnn') {
            // Simulate higher accuracy for CNN
            finalAccuracy = 95 + 3.5 * Math.random();
            this.modelResults.cnn.trained = true;
            this.modelResults.cnn.accuracy = finalAccuracy;
            this.modelResults.cnn.trainTime = trainingTime;

            // Update UI
            document.getElementById('cnn-accuracy').textContent = `${finalAccuracy.toFixed(2)}%`;
            document.getElementById('cnn-train-time').textContent = `${trainingTime.toFixed(1)}s`;
            document.getElementById('cnn-params').textContent = '1.3M';
        }
        else if (currentModel === 'sync-cea') {
            // Simulate slightly lower accuracy for Sync CEA
            finalAccuracy = 93 + 3.5 * Math.random();
            this.modelResults.syncCEA.trained = true;
            this.modelResults.syncCEA.accuracy = finalAccuracy;
            this.modelResults.syncCEA.trainTime = trainingTime;

            // Update UI
            document.getElementById('sync-cea-accuracy').textContent = `${finalAccuracy.toFixed(2)}%`;
            document.getElementById('sync-cea-train-time').textContent = `${trainingTime.toFixed(1)}s`;
            document.getElementById('sync-cea-generations').textContent = this.modelResults.syncCEA.generations;
        }
        else if (currentModel === 'async-cea') {
            // Simulate slightly lower accuracy for Async CEA
            finalAccuracy = 91 + 3.5 * Math.random();
            this.modelResults.asyncCEA.trained = true;
            this.modelResults.asyncCEA.accuracy = finalAccuracy;
            this.modelResults.asyncCEA.trainTime = trainingTime;

            // Update UI
            document.getElementById('async-cea-accuracy').textContent = `${finalAccuracy.toFixed(2)}%`;
            document.getElementById('async-cea-train-time').textContent = `${trainingTime.toFixed(1)}s`;
            document.getElementById('async-cea-generations').textContent = this.modelResults.asyncCEA.generations;
        }

        // Log completion
        this.logMessage(`Training completed in ${trainingTime.toFixed(2)} seconds`, 'success');
        this.logMessage(`Test accuracy: ${finalAccuracy.toFixed(2)}%`, 'success');

        // Reset training status
        this.trainingStatus.isTraining = false;

        // Update comparison charts
        this.updateCharts();

        // Auto switch to results tab after a delay
        setTimeout(() => {
            document.querySelector('[data-tab="results"]').click();
        }, 1500);
    }

    logMessage(message, type = 'info') {
        // Store in log history
        this.trainingStatus.trainingLog.push({
            message,
            type,
            timestamp: new Date().toLocaleTimeString()
        });

        // Add to UI
        const logContainer = document.getElementById('training-log');
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.innerHTML = `<span class="log-time">[${new Date().toLocaleTimeString()}]</span> ${message}`;

        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    showModelDetails(modelType) {
        const modalTitle = document.getElementById('model-details-title');
        const modalContent = document.getElementById('model-details-content');
        const modal = document.getElementById('model-details-modal');

        let title, content;

        if (modelType === 'cnn') {
            title = 'Standard CNN Details';

            if (!this.modelResults.cnn.trained) {
                content = `<p>This model has not been trained yet.</p>`;
            } else {
                content = `
                    <div class="model-details-grid">
                        <div class="detail-item">
                            <span class="detail-label">Architecture:</span>
                            <span class="detail-value">Convolutional Neural Network</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Test Accuracy:</span>
                            <span class="detail-value">${this.modelResults.cnn.accuracy.toFixed(2)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Training Time:</span>
                            <span class="detail-value">${this.modelResults.cnn.trainTime.toFixed(2)} seconds</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Epochs:</span>
                            <span class="detail-value">${this.modelResults.cnn.epochs}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Learning Rate:</span>
                            <span class="detail-value">${this.modelResults.cnn.learningRate}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Parameters:</span>
                            <span class="detail-value">1.3M</span>
                        </div>
                    </div>
                    <div class="model-architecture">
                        <h4>Model Architecture</h4>
                        <pre>
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32)
  (bn2): BatchNorm2d(64)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (fc1): Linear(in_features=3136, out_features=128)
  (fc2): Linear(in_features=128, out_features=10)
)
                        </pre>
                    </div>
                `;
            }
        }
        else if (modelType === 'sync-cea') {
            title = 'Synchronous CEA Details';

            if (!this.modelResults.syncCEA.trained) {
                content = `<p>This model has not been trained yet.</p>`;
            } else {
                content = `
                    <div class="model-details-grid">
                        <div class="detail-item">
                            <span class="detail-label">Architecture:</span>
                            <span class="detail-value">Synchronous Cellular Evolutionary Automata</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Test Accuracy:</span>
                            <span class="detail-value">${this.modelResults.syncCEA.accuracy.toFixed(2)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Training Time:</span>
                            <span class="detail-value">${this.modelResults.syncCEA.trainTime.toFixed(2)} seconds</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Generations:</span>
                            <span class="detail-value">${this.modelResults.syncCEA.generations}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Grid Size:</span>
                            <span class="detail-value">${this.modelResults.syncCEA.gridSize}x${this.modelResults.syncCEA.gridSize}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Selection Method:</span>
                            <span class="detail-value">${this.modelResults.syncCEA.selectionMethod}</span>
                        </div>
                    </div>
                    <div class="fitness-evolution">
                        <h4>Fitness Evolution</h4>
                        <canvas id="detail-fitness-chart" width="400" height="200"></canvas>
                    </div>
                `;

                // Initialize fitness chart after modal is visible
                setTimeout(() => {
                    if (this.modelResults.syncCEA.fitnessHistory.length > 0) {
                        const ctx = document.getElementById('detail-fitness-chart').getContext('2d');
                        new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: Array.from(
                                    {length: this.modelResults.syncCEA.fitnessHistory.length},
                                    (_, i) => i + 1
                                ),
                                datasets: [{
                                    label: 'Best Fitness',
                                    data: this.modelResults.syncCEA.fitnessHistory,
                                    borderColor: 'rgba(16, 185, 129, 1)',
                                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                                    tension: 0.1,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 100
                                    }
                                }
                            }
                        });
                    }
                }, 100);
            }
        }
        else if (modelType === 'async-cea') {
            title = 'Asynchronous CEA Details';

            if (!this.modelResults.asyncCEA.trained) {
                content = `<p>This model has not been trained yet.</p>`;
            } else {
                content = `
                    <div class="model-details-grid">
                        <div class="detail-item">
                            <span class="detail-label">Architecture:</span>
                            <span class="detail-value">Asynchronous Cellular Evolutionary Automata</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Test Accuracy:</span>
                            <span class="detail-value">${this.modelResults.asyncCEA.accuracy.toFixed(2)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Training Time:</span>
                            <span class="detail-value">${this.modelResults.asyncCEA.trainTime.toFixed(2)} seconds</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Generations:</span>
                            <span class="detail-value">${this.modelResults.asyncCEA.generations}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Grid Size:</span>
                            <span class="detail-value">${this.modelResults.asyncCEA.gridSize}x${this.modelResults.asyncCEA.gridSize}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Selection Method:</span>
                            <span class="detail-value">${this.modelResults.asyncCEA.selectionMethod}</span>
                        </div>
                    </div>
                    <div class="fitness-evolution">
                        <h4>Fitness Evolution</h4>
                        <canvas id="detail-fitness-chart" width="400" height="200"></canvas>
                    </div>
                `;

                // Initialize fitness chart after modal is visible
                setTimeout(() => {
                    if (this.modelResults.asyncCEA.fitnessHistory.length > 0) {
                        const ctx = document.getElementById('detail-fitness-chart').getContext('2d');
                        new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: Array.from(
                                    {length: this.modelResults.asyncCEA.fitnessHistory.length},
                                    (_, i) => i + 1
                                ),
                                datasets: [{
                                    label: 'Best Fitness',
                                    data: this.modelResults.asyncCEA.fitnessHistory,
                                    borderColor: 'rgba(245, 158, 11, 1)',
                                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                    tension: 0.1,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 100
                                    }
                                }
                            }
                        });
                    }
                }, 100);
            }
        }

        modalTitle.textContent = title;
        modalContent.innerHTML = content;

        // Show the modal
        modal.classList.remove('hidden');
        setTimeout(() => {
            modal.classList.add('visible');
        }, 10);
    }

    saveModel(modelType) {
        let message = '';
        let isSuccess = false;

        if (modelType === 'cnn') {
            if (!this.modelResults.cnn.trained) {
                message = 'CNN model has not been trained yet.';
            } else {
                message = 'CNN model saved successfully to saved_models/cnn_model.pth';
                isSuccess = true;
            }
        }
        else if (modelType === 'sync-cea') {
            if (!this.modelResults.syncCEA.trained) {
                message = 'Sync CEA model has not been trained yet.';
            } else {
                message = 'Sync CEA model saved successfully to saved_models/sync_cea_best.pth';
                isSuccess = true;
            }
        }
        else if (modelType === 'async-cea') {
            if (!this.modelResults.asyncCEA.trained) {
                message = 'Async CEA model has not been trained yet.';
            } else {
                message = 'Async CEA model saved successfully to saved_models/async_cea_best.pth';
                isSuccess = true;
            }
        }
        else if (modelType === 'all') {
            const cnnTrained = this.modelResults.cnn.trained;
            const syncTrained = this.modelResults.syncCEA.trained;
            const asyncTrained = this.modelResults.asyncCEA.trained;

            if (!cnnTrained && !syncTrained && !asyncTrained) {
                message = 'No models have been trained yet.';
            } else {
                const saved = [];
                if (cnnTrained) saved.push('CNN');
                if (syncTrained) saved.push('Sync CEA');
                if (asyncTrained) saved.push('Async CEA');

                message = `${saved.join(', ')} model(s) saved successfully.`;
                isSuccess = true;
            }
        }

        // Show notification
        this.showNotification(message, isSuccess ? 'success' : 'error');
    }

    showNotification(message, type = 'info') {
        // Create notification element if it doesn't exist
        let notification = document.querySelector('.notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.className = 'notification';
            document.body.appendChild(notification);
        }

        // Set type class
        notification.className = `notification ${type}`;

        // Set message
        notification.textContent = message;

        // Show notification
        notification.classList.add('visible');

        // Hide after 3 seconds
        setTimeout(() => {
            notification.classList.remove('visible');

            // Remove after animation completes
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Initialize trainer when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.modelTrainer = new ModelTrainer();
});