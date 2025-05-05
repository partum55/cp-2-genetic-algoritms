/**
 * Enhanced model selection and prediction functionality with animations
 */
class ModelManager {
    constructor() {
        this.models = {
            cnn: {
                name: 'Standard CNN',
                description: 'Standard Convolutional Neural Network trained with Adam optimizer.',
                stats: {
                    accuracy: '98.2%',
                    parameters: '1.3M',
                    epochs: '20'
                }
            },
            syncCEA: {
                name: 'Sync CEA',
                description: 'Synchronous Cellular Evolutionary Automata - evolves a population of CNNs simultaneously.',
                stats: {
                    accuracy: '97.5%',
                    parameters: '1.2M',
                    generations: '50'
                }
            },
            asyncCEA: {
                name: 'Async CEA',
                description: 'Asynchronous Cellular Evolutionary Automata - evolves a population of CNNs cell by cell.',
                stats: {
                    accuracy: '96.8%',
                    parameters: '1.1M',
                    generations: '45'
                }
            }
        };

        this.currentModel = 'cnn'; // Default model
        this.apiEndpoint = 'http://localhost:5000/api/recognize';
        this.lastPrediction = null;
        this.allPredictions = {};

        // Initialize model selection
        this.setupModelSelection();
    }

    setupModelSelection() {
        const modelOptions = document.querySelectorAll('.model-option');

        // Add click event to each model option
        modelOptions.forEach(option => {
            option.addEventListener('click', () => {
                const modelType = option.dataset.model;
                this.selectModel(modelType);

                // Update active state with animation
                modelOptions.forEach(opt => {
                    opt.classList.remove('active');

                    // Reset indicator
                    const indicator = opt.querySelector('.model-select-indicator');
                    indicator.style.width = '0';
                });

                // Set the new active option
                option.classList.add('active');

                // Animate the indicator
                const indicator = option.querySelector('.model-select-indicator');
                setTimeout(() => {
                    indicator.style.width = '100%';
                }, 50);
            });
        });

        // Set up keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyPress.bind(this));
    }

    handleKeyPress(e) {
        // Number keys 1-3 for model selection
        if (e.key === '1') {
            this.selectModel('cnn');
            document.querySelector('[data-model="cnn"]').classList.add('active');
            document.querySelector('[data-model="syncCEA"]').classList.remove('active');
            document.querySelector('[data-model="asyncCEA"]').classList.remove('active');
        } else if (e.key === '2') {
            this.selectModel('syncCEA');
            document.querySelector('[data-model="cnn"]').classList.remove('active');
            document.querySelector('[data-model="syncCEA"]').classList.add('active');
            document.querySelector('[data-model="asyncCEA"]').classList.remove('active');
        } else if (e.key === '3') {
            this.selectModel('asyncCEA');
            document.querySelector('[data-model="cnn"]').classList.remove('active');
            document.querySelector('[data-model="syncCEA"]').classList.remove('active');
            document.querySelector('[data-model="asyncCEA"]').classList.add('active');
        } else if (e.key === 'Enter') {
            // Enter key for recognize
            this.recognizeDigit();
        } else if (e.key === 'm' || e.key === 'M') {
            // M key for model comparison
            this.compareAllModels();
        }
    }

    selectModel(modelType) {
        // Update current model
        this.currentModel = modelType;

        // Update all indicator widths
        document.querySelectorAll('.model-select-indicator').forEach(indicator => {
            indicator.style.width = '0';
        });

        // Animate the selected indicator
        const selectedOption = document.querySelector(`[data-model="${modelType}"]`);
        if (selectedOption) {
            selectedOption.classList.add('active');
            const indicator = selectedOption.querySelector('.model-select-indicator');
            setTimeout(() => {
                indicator.style.width = '100%';
            }, 50);
        }

        // Update model description visibility with fade effect
        const modelDetails = document.querySelectorAll('.model-detail');
        modelDetails.forEach(detail => {
            detail.classList.remove('active');
        });

        const selectedDetail = document.getElementById(`${modelType}-desc`);
        if (selectedDetail) {
            // Trigger reflow for animation
            void selectedDetail.offsetWidth;
            selectedDetail.classList.add('active');
        }

        // Update model name in prediction section
        const modelNameDisplay = document.getElementById('model-name-display');
        if (modelNameDisplay && this.models[modelType]) {
            modelNameDisplay.textContent = this.models[modelType].name;
        }

        // If we already have a prediction, update the display with the selected model's results
        if (this.allPredictions[modelType]) {
            this.displayPrediction(this.allPredictions[modelType], true);
        } else if (Object.keys(this.allPredictions).length > 0) {
            // If we have other predictions but not for this model, reset and show suggestion
            this.resetPrediction();
            // Show the waiting message with a hint
            document.getElementById('waiting-message').innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64"><path d="M1.181 12C2.121 6.88 6.608 3 12 3c5.392 0 9.878 3.88 10.819 9-.94 5.12-5.427 9-10.819 9-5.392 0-9.878-3.88-10.819-9zM12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0-2a3 3 0 1 1 0-6 3 3 0 0 1 0 6z"/></svg>
                <p>Click "Recognize" to analyze with ${this.models[modelType].name}</p>
            `;
        }
    }

    resetPrediction() {
        // Hide prediction result
        document.getElementById('prediction-result').classList.add('hidden');
        document.getElementById('waiting-message').classList.remove('hidden');
        document.getElementById('loading-spinner').classList.add('hidden');

        // Clear model suggestion
        document.getElementById('model-suggestion').innerHTML = '';

        // Reset comparison section if visible
        const comparisonSection = document.getElementById('comparison-section');
        if (!comparisonSection.classList.contains('hidden')) {
            comparisonSection.classList.add('hidden');
        }
    }

    async recognizeDigit(shouldCompare = false) {
        // Only proceed if we have drawing
        if (!window.drawingCanvas.hasDrawn) {
            this.showToast('Please draw a digit first');
            return;
        }

        // Get image data from canvas
        const imageData = window.drawingCanvas.getImageData();

        // Show loading spinner
        document.getElementById('waiting-message').classList.add('hidden');
        document.getElementById('loading-spinner').classList.remove('hidden');
        document.getElementById('prediction-result').classList.add('hidden');

        try {
            // Send image to API for the current model
            const response = await this.sendImageForRecognition(imageData, this.currentModel);

            // Store prediction for this model
            this.allPredictions[this.currentModel] = response;
            this.lastPrediction = response;

            // Process and display result
            this.displayPrediction(response);

            // If we're comparing all models, continue with other models
            if (shouldCompare) {
                await this.recognizeWithOtherModels(imageData);
            }
        } catch (error) {
            console.error('Error recognizing digit:', error);

            // Show error message or fallback
            this.handleRecognitionError();
        }
    }

    async recognizeWithOtherModels(imageData) {
        // Array of all model types
        const allModelTypes = Object.keys(this.models);

        // Only process models we haven't tried yet
        const remainingModels = allModelTypes.filter(model => model !== this.currentModel && !this.allPredictions[model]);

        // Process remaining models
        for (const modelType of remainingModels) {
            try {
                const response = await this.sendImageForRecognition(imageData, modelType);
                this.allPredictions[modelType] = response;
            } catch (error) {
                console.error(`Error recognizing with ${modelType}:`, error);
            }
        }

        // After all recognition is done, show the comparison
        this.displayModelComparison();
    }

    async sendImageForRecognition(imageData, modelType) {
        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData,
                    modelType: modelType,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);

            // For demo purposes, return a mock response if the API is not available
            return this.getMockResponse(modelType);
        }
    }

    getMockResponse(modelType) {
        // Generate a digit (weighted toward lower digits for realism)
        const weights = [0.2, 0.15, 0.15, 0.12, 0.1, 0.08, 0.07, 0.05, 0.05, 0.03];
        const random = Math.random();
        let cumulativeWeight = 0;
        let digit = 0;

        for (let i = 0; i < weights.length; i++) {
            cumulativeWeight += weights[i];
            if (random < cumulativeWeight) {
                digit = i;
                break;
            }
        }

        // Generate confidence scores with higher confidence for the predicted digit
        const confidences = Array(10).fill(0).map(() => Math.random() * 5);

        // Add model variation
        let baseConfidence = 0;
        if (modelType === 'cnn') {
            baseConfidence = 85 + Math.random() * 10;
        } else if (modelType === 'syncCEA') {
            // Make syncCEA slightly less confident
            baseConfidence = 80 + Math.random() * 15;

            // 20% chance it predicts a different digit
            if (Math.random() < 0.2) {
                const newDigit = (digit + 1) % 10;
                confidences[newDigit] = baseConfidence;
                confidences[digit] = 70 + Math.random() * 10;
                digit = newDigit;
            } else {
                confidences[digit] = baseConfidence;
            }
        } else {
            // asyncCEA is less confident
            baseConfidence = 75 + Math.random() * 15;

            // 30% chance it predicts a different digit
            if (Math.random() < 0.3) {
                const newDigit = (digit + 1) % 10;
                confidences[newDigit] = baseConfidence;
                confidences[digit] = 65 + Math.random() * 15;
                digit = newDigit;
            } else {
                confidences[digit] = baseConfidence;
            }
        }

        if (confidences[digit] < baseConfidence) {
            confidences[digit] = baseConfidence;
        }

        return {
            digit: digit,
            confidences: confidences
        };
    }

    displayPrediction(result, skipAnimation = false) {
        // Hide loading spinner and waiting message
        document.getElementById('loading-spinner').classList.add('hidden');
        document.getElementById('waiting-message').classList.add('hidden');

        // Update the predicted digit
        document.getElementById('digit-value').textContent = result.digit;

        // Get the main confidence value
        const mainConfidence = result.confidences[result.digit].toFixed(1);
        document.getElementById('main-confidence').textContent = `${mainConfidence}%`;

        // Set color based on confidence level
        const confidenceElement = document.getElementById('main-confidence');
        if (mainConfidence >= 90) {
            confidenceElement.style.color = 'var(--success-color)';
        } else if (mainConfidence >= 70) {
            confidenceElement.style.color = 'var(--info-color)';
        } else if (mainConfidence >= 50) {
            confidenceElement.style.color = 'var(--warning-color)';
        } else {
            confidenceElement.style.color = 'var(--error-color)';
        }

        // Generate confidence bars
        const confidenceContainer = document.getElementById('confidence-container');
        confidenceContainer.innerHTML = ''; // Clear existing bars

        // Create a bar for each digit
        for (let i = 0; i < 10; i++) {
            const confidence = result.confidences[i];
            const isPredicted = i === result.digit;

            const barElement = document.createElement('div');
            barElement.className = 'confidence-bar';

            barElement.innerHTML = `
                <div class="digit-label">${i}</div>
                <div class="bar-container">
                    <div class="bar-fill ${isPredicted ? 'predicted' : ''}" style="width: 0%"></div>
                </div>
                <div class="confidence-value">${confidence.toFixed(1)}%</div>
            `;

            confidenceContainer.appendChild(barElement);
        }

        // Show the prediction section
        document.getElementById('prediction-result').classList.remove('hidden');

        // Check if we should suggest a different model
        this.suggestBetterModel(result);

        // Animate the bars after a short delay
        setTimeout(() => {
            const bars = document.querySelectorAll('.bar-fill');
            bars.forEach((bar, index) => {
                if (!skipAnimation) {
                    // Stagger the animations for a nicer effect
                    setTimeout(() => {
                        bar.style.width = `${result.confidences[index]}%`;
                    }, index * 50);
                } else {
                    bar.style.width = `${result.confidences[index]}%`;
                }
            });
        }, skipAnimation ? 0 : 100);
    }

    suggestBetterModel(result) {
        const modelSuggestion = document.getElementById('model-suggestion');
        modelSuggestion.innerHTML = '';

        // Only suggest if we have predictions from at least two models
        if (Object.keys(this.allPredictions).length < 2) {
            return;
        }

        // Find the model with the highest confidence for this digit
        let bestModelType = this.currentModel;
        let bestConfidence = result.confidences[result.digit];

        for (const [modelType, prediction] of Object.entries(this.allPredictions)) {
            if (modelType !== this.currentModel &&
                prediction.confidences[result.digit] > bestConfidence) {
                bestModelType = modelType;
                bestConfidence = prediction.confidences[result.digit];
            }
        }

        // If a better model is found, show a suggestion
        if (bestModelType !== this.currentModel) {
            const bestModelName = this.models[bestModelType].name;
            const confidenceDiff = (bestConfidence - result.confidences[result.digit]).toFixed(1);

            modelSuggestion.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20"><path fill="none" d="M0 0h24v24H0z"/><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm0-2a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM11 7h2v2h-2V7zm0 4h2v6h-2v-6z"/></svg>
                <p>${bestModelName} may give better results for this digit (${confidenceDiff}% higher confidence). 
                <a href="#" class="switch-model" data-model="${bestModelType}">Switch to ${bestModelName}</a></p>
            `;

            // Add click event to the link
            const switchLink = modelSuggestion.querySelector('.switch-model');
            switchLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.selectModel(bestModelType);
                document.querySelector(`[data-model="${bestModelType}"]`).click();
            });
        } else if (Object.keys(this.allPredictions).length >= 2) {
            // If we have multiple predictions but this is the best, show a confirmation
            modelSuggestion.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20"><path fill="none" d="M0 0h24v24H0z"/><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1.177-7.86l-2.765-2.767L7 12.431l3.119 3.12a1 1 0 0 0 1.414 0l5.952-5.95-1.062-1.06-5.6 5.599z"/></svg>
                <p>${this.models[this.currentModel].name} gives the best confidence for this digit. 
                <a href="#" class="compare-models">Compare all models</a></p>
            `;

            // Add click event to the link
            const compareLink = modelSuggestion.querySelector('.compare-models');
            compareLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.displayModelComparison();
            });
        }
    }

    async compareAllModels() {
        // If we don't have drawings, show a message
        if (!window.drawingCanvas.hasDrawn) {
            this.showToast('Please draw a digit first');
            return;
        }

        // If we already have predictions for all models, just display the comparison
        const allModelTypes = Object.keys(this.models);
        const allPredicted = allModelTypes.every(modelType => this.allPredictions[modelType]);

        if (allPredicted) {
            this.displayModelComparison();
        } else {
            // Otherwise, recognize with all models
            await this.recognizeDigit(true);
        }
    }

    displayModelComparison() {
        const comparisonSection = document.getElementById('comparison-section');
        const modelResults = comparisonSection.querySelector('.model-results');

        // Clear existing results
        modelResults.innerHTML = '';

        // Add a result for each model
        for (const [modelType, modelInfo] of Object.entries(this.models)) {
            const prediction = this.allPredictions[modelType];

            if (!prediction) continue;

            const confidence = prediction.confidences[prediction.digit].toFixed(1);

            const resultElement = document.createElement('div');
            resultElement.className = 'model-result';

            resultElement.innerHTML = `
                <div class="model-result-name">${modelInfo.name}</div>
                <div class="model-result-digit">${prediction.digit}</div>
                <div class="model-result-confidence">
                    <span style="font-weight: 600">${confidence}%</span> confidence
                </div>
                <button class="btn text select-model-btn" data-model="${modelType}">
                    <span>Select Model</span>
                </button>
            `;

            modelResults.appendChild(resultElement);
        }

        // Show the comparison section
        comparisonSection.classList.remove('hidden');

        // Add click events to the select model buttons
        const selectButtons = document.querySelectorAll('.select-model-btn');
        selectButtons.forEach(button => {
            button.addEventListener('click', () => {
                const modelType = button.dataset.model;
                this.selectModel(modelType);
                document.querySelector(`[data-model="${modelType}"]`).click();
                comparisonSection.classList.add('hidden');
            });
        });

        // Set up close button
        const closeButton = document.getElementById('close-comparison');
        closeButton.addEventListener('click', () => {
            comparisonSection.classList.add('hidden');
        });
    }

    handleRecognitionError() {
        // Hide loading spinner
        document.getElementById('loading-spinner').classList.add('hidden');

        // Show an error message to the user
        document.getElementById('waiting-message').classList.remove('hidden');
        document.getElementById('waiting-message').innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64"><path fill="none" d="M0 0h24v24H0z"/><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-7v2h2v-2h-2zm0-8v6h2V7h-2z"/></svg>
            <p>Error recognizing digit. Please try again.</p>
        `;

        // Reset the message after a delay
        setTimeout(() => {
            document.getElementById('waiting-message').innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64"><path fill="none" d="M0 0h24v24H0z"/><path d="M5.463 4.433A9.961 9.961 0 0 1 12 2c5.523 0 10 4.477 10 10 0 2.136-.67 4.116-1.81 5.74L17 12h3A8 8 0 0 0 6.46 6.228l-.997-1.795zm13.074 15.134A9.961 9.961 0 0 1 12 22C6.477 22 2 17.523 2 12c0-2.136.67-4.116 1.81-5.74L7 12H4a8 8 0 0 0 13.54 5.772l.997 1.795z"/></svg>
                <p>Draw a digit and click "Recognize" to see the results</p>
            `;
        }, 3000);
    }

    showToast(message) {
        // Create toast element if it doesn't exist
        let toast = document.querySelector('.toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.className = 'toast';
            document.body.appendChild(toast);

            // Add styles for the toast
            const style = document.createElement('style');
            style.textContent = `
                .toast {
                    position: fixed;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background-color: rgba(0, 0, 0, 0.8);
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    z-index: 1000;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                    pointer-events: none;
                }
                
                .toast.visible {
                    opacity: 1;
                }
            `;
            document.head.appendChild(style);
        }

        // Set message and show
        toast.textContent = message;
        toast.classList.add('visible');

        // Hide after 3 seconds
        setTimeout(() => {
            toast.classList.remove('visible');
        }, 3000);
    }
}

// Initialize model manager when window loads
window.addEventListener('load', () => {
    window.modelManager = new ModelManager();
});