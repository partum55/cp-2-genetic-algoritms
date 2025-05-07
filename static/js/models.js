/**
 * Enhanced model selection and prediction functionality with support for both
 * single model and comparison modes
 */
class ModelManager {
    constructor() {
        // Default model information, will be replaced with real metrics if available
        this.models = {
            cnn: {
                name: 'Standard CNN',
                description: 'Standard Convolutional Neural Network trained with Adam optimizer.',
                stats: {
                    accuracy: 'Loading...',
                    parameters: '1.3M',
                    epochs: 'Loading...'
                }
            },
            syncCEA: {
                name: 'Sync CEA',
                description: 'Synchronous Cellular Evolutionary Automata - evolves a population of CNNs simultaneously.',
                stats: {
                    accuracy: 'Loading...',
                    parameters: '1.2M',
                    generations: 'Loading...'
                }
            },
            asyncCEA: {
                name: 'Async CEA',
                description: 'Asynchronous Cellular Evolutionary Automata - evolves a population of CNNs cell by cell.',
                stats: {
                    accuracy: 'Loading...',
                    parameters: '1.1M',
                    generations: 'Loading...'
                }
            }
        };

        this.currentModel = 'cnn'; // Default model
        this.apiEndpoint = 'http://localhost:5000/api/recognize';
        this.lastPrediction = null;
        this.allPredictions = {};
        this.isComparisonMode = false;

        // Load real metrics from server
        this.loadRealMetrics();

        // Initialize model selection
        this.setupModelSelection();
        this.setupRecognizeButtons();
    }

    loadRealMetrics() {
        fetch('http://localhost:5000/api/model_metrics')
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('Failed to load model metrics');
            })
            .then(metrics => {
                // Update models with real metrics if available
                if (metrics && Object.keys(metrics).length > 0) {
                    this.models = metrics;

                    // Update the display on the page
                    this.updateModelMetricsDisplay();
                    console.log('Loaded real model metrics');
                }
            })
            .catch(error => {
                console.warn('Using default metrics:', error);
            });
    }

    updateModelMetricsDisplay() {
        // Update stats display for each model on the page
        for (const [modelType, modelData] of Object.entries(this.models)) {
            // Update statistics in model details section
            const modelStats = document.querySelectorAll(`#${modelType}-desc .stat-value`);
            if (modelStats.length >= 3) {
                modelStats[0].textContent = modelData.stats.accuracy;
                modelStats[1].textContent = modelData.stats.parameters;

                // Check if it's epochs or generations
                if ('epochs' in modelData.stats) {
                    modelStats[2].textContent = modelData.stats.epochs;
                } else if ('generations' in modelData.stats) {
                    modelStats[2].textContent = modelData.stats.generations;
                }
            }
        }

        // If we're already on the comparison page and have predictions, update them too
        if (this.isComparisonMode && Object.keys(this.allPredictions).length > 0) {
            this.displayComparisonResults();
        }
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
                    if (indicator) indicator.style.width = '0';
                });

                // Set the new active option
                option.classList.add('active');

                // Animate the indicator
                const indicator = option.querySelector('.model-select-indicator');
                if (indicator) {
                    setTimeout(() => {
                        indicator.style.width = '100%';
                    }, 50);
                }
            });
        });

        // Set up keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyPress.bind(this));
    }

    setupRecognizeButtons() {
        // Set up single model recognize button
        const singleRecognizeButton = document.getElementById('single-recognize-button');
        if (singleRecognizeButton) {
            singleRecognizeButton.addEventListener('click', () => {
                this.recognizeSingleModel();
            });
        }

        // Set up comparison recognize button
        const comparisonRecognizeButton = document.getElementById('comparison-recognize-button');
        if (comparisonRecognizeButton) {
            comparisonRecognizeButton.addEventListener('click', () => {
                this.recognizeAllModels();
            });
        }
    }

    handleKeyPress(e) {
        // Get the current visible container
        const singleContainer = document.getElementById('single-model-container');
        const comparisonContainer = document.getElementById('comparison-container');

        // Only handle model selection shortcuts if in single mode and it's visible
        if (!singleContainer.classList.contains('hidden')) {
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
            }
        }

        // Handle Enter key for recognize in both modes
        if (e.key === 'Enter') {
            if (!singleContainer.classList.contains('hidden')) {
                this.recognizeSingleModel();
            } else if (!comparisonContainer.classList.contains('hidden')) {
                this.recognizeAllModels();
            }
        }
    }

    setComparisonMode(isComparison) {
        this.isComparisonMode = isComparison;
        this.allPredictions = {}; // Reset predictions when changing modes
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
        const modelNameDisplay = document.getElementById('single-model-name-display');
        if (modelNameDisplay && this.models[modelType]) {
            modelNameDisplay.textContent = this.models[modelType].name;
        }

        // If we already have a prediction, update the display with the selected model's results
        if (this.allPredictions[modelType]) {
            this.displaySinglePrediction(this.allPredictions[modelType], true);
        } else if (Object.keys(this.allPredictions).length > 0) {
            // If we have other predictions but not for this model, reset and show suggestion
            this.resetSinglePrediction();
            // Show the waiting message with a hint
            document.getElementById('single-waiting-message').innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64"><path d="M1.181 12C2.121 6.88 6.608 3 12 3c5.392 0 9.878 3.88 10.819 9-.94 5.12-5.427 9-10.819 9-5.392 0-9.878-3.88-10.819-9zM12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0-2a3 3 0 1 1 0-6 3 3 0 0 1 0 6z"/></svg>
                <p>Click "Recognize" to analyze with ${this.models[modelType].name}</p>
            `;
        }
    }

    resetSinglePrediction() {
        // Hide prediction result
        document.getElementById('single-prediction-result').classList.add('hidden');
        document.getElementById('single-waiting-message').classList.remove('hidden');
        document.getElementById('single-loading-spinner').classList.add('hidden');
    }

    async recognizeSingleModel() {
        if (!window.singleDrawingCanvas || !window.singleDrawingCanvas.hasDrawn) {
            this.showToast('Please draw a digit first');
            return;
        }

        // Get image data from canvas
        const imageData = window.singleDrawingCanvas.getImageData();

        // Show loading spinner
        document.getElementById('single-waiting-message').classList.add('hidden');
        document.getElementById('single-loading-spinner').classList.remove('hidden');
        document.getElementById('single-prediction-result').classList.add('hidden');

        try {
            // Send image to API for the current model
            const response = await this.sendImageForRecognition(imageData, this.currentModel);

            // Store prediction for this model
            this.allPredictions[this.currentModel] = response;
            this.lastPrediction = response;

            // Process and display result
            this.displaySinglePrediction(response);
        } catch (error) {
            console.error('Error recognizing digit:', error);
            this.handleRecognitionError('single');
        }
    }

    async recognizeAllModels() {
        if (!window.comparisonDrawingCanvas || !window.comparisonDrawingCanvas.hasDrawn) {
            this.showToast('Please draw a digit first');
            return;
        }

        // Get image data from canvas
        const imageData = window.comparisonDrawingCanvas.getImageData();

        // Show loading spinner
        document.getElementById('comparison-waiting-message').classList.add('hidden');
        document.getElementById('comparison-loading-spinner').classList.remove('hidden');
        document.getElementById('comparison-prediction-result').classList.add('hidden');

        // Hide comparison results section initially
        const comparisonResultsSection = document.getElementById('comparison-results-section');
        if (comparisonResultsSection) {
            comparisonResultsSection.classList.add('hidden');
        }

        // Clear previous predictions
        this.allPredictions = {};

        try {
            // Process all models
            const modelTypes = Object.keys(this.models);

            for (const modelType of modelTypes) {
                const response = await this.sendImageForRecognition(imageData, modelType);
                this.allPredictions[modelType] = response;
            }

            // Hide loading spinner and show scrolling message
            document.getElementById('comparison-loading-spinner').classList.add('hidden');
            document.getElementById('comparison-prediction-result').classList.remove('hidden');

            // Display comparison results
            this.displayComparisonResults();

        } catch (error) {
            console.error('Error recognizing digit with all models:', error);
            this.handleRecognitionError('comparison');
        }
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
        // Get real accuracy from model metrics for more realistic demo
        let baseAccuracy = 0;
        try {
            const accuracyStr = this.models[modelType].stats.accuracy;
            baseAccuracy = parseFloat(accuracyStr.replace('%', ''));
            if (isNaN(baseAccuracy)) {
                baseAccuracy = modelType === 'cnn' ? 95 :
                               modelType === 'syncCEA' ? 90 : 85;
            }
        } catch (e) {
            baseAccuracy = modelType === 'cnn' ? 95 :
                          modelType === 'syncCEA' ? 90 : 85;
        }

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

        // Add model variation based on actual metrics
        let confidenceLevel = 0;

        if (modelType === 'cnn') {
            confidenceLevel = baseAccuracy - 10 + Math.random() * 10; // Slightly below accuracy
        } else if (modelType === 'syncCEA') {
            confidenceLevel = baseAccuracy - 15 + Math.random() * 15;

            // 20% chance it predicts a different digit
            if (Math.random() < 0.2) {
                const newDigit = (digit + 1) % 10;
                confidences[newDigit] = confidenceLevel;
                confidences[digit] = confidenceLevel - 10 + Math.random() * 10;
                digit = newDigit;
            } else {
                confidences[digit] = confidenceLevel;
            }
        } else {
            confidenceLevel = baseAccuracy - 20 + Math.random() * 15;

            // 30% chance it predicts a different digit
            if (Math.random() < 0.3) {
                const newDigit = (digit + 1) % 10;
                confidences[newDigit] = confidenceLevel;
                confidences[digit] = confidenceLevel - 15 + Math.random() * 15;
                digit = newDigit;
            } else {
                confidences[digit] = confidenceLevel;
            }
        }

        if (confidences[digit] < confidenceLevel) {
            confidences[digit] = confidenceLevel;
        }

        return {
            digit: digit,
            confidences: confidences
        };
    }

    displaySinglePrediction(result, skipAnimation = false) {
        // Hide loading spinner and waiting message
        document.getElementById('single-loading-spinner').classList.add('hidden');
        document.getElementById('single-waiting-message').classList.add('hidden');

        // Update the predicted digit with proper formatting
        const digitElement = document.getElementById('single-digit-value');
        digitElement.textContent = result.digit;

        // Ensure digit is perfectly centered
        digitElement.style.display = 'flex';
        digitElement.style.alignItems = 'center';
        digitElement.style.justifyContent = 'center';
        digitElement.style.width = '100%';
        digitElement.style.height = '100%';

        // Get the main confidence value
        const mainConfidence = result.confidences[result.digit].toFixed(1);
        document.getElementById('single-main-confidence').textContent = `${mainConfidence}%`;

        // Set color based on confidence level
        const confidenceElement = document.getElementById('single-main-confidence');
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
        const confidenceContainer = document.getElementById('single-confidence-container');
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
        document.getElementById('single-prediction-result').classList.remove('hidden');

        // Animate the bars after a short delay
        setTimeout(() => {
            const bars = document.querySelectorAll('#single-confidence-container .bar-fill');
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

    displayComparisonResults() {
        // Show the comparison section
        const comparisonSection = document.getElementById('comparison-results-section');
        comparisonSection.classList.remove('hidden');

        // Get the container for model results
        const modelResultsContainer = document.getElementById('comparison-model-results');
        modelResultsContainer.innerHTML = ''; // Clear existing results

        // Find the best model (highest confidence for predicted digit)
        let bestModelType = null;
        let bestConfidence = -1;

        for (const [modelType, prediction] of Object.entries(this.allPredictions)) {
            const confidence = prediction.confidences[prediction.digit];
            if (confidence > bestConfidence) {
                bestConfidence = confidence;
                bestModelType = modelType;
            }
        }

        // Create a card for each model
        for (const [modelType, modelInfo] of Object.entries(this.models)) {
            const prediction = this.allPredictions[modelType];
            if (!prediction) continue;

            const confidence = prediction.confidences[prediction.digit].toFixed(1);
            const isBestModel = modelType === bestModelType;

            const resultCard = document.createElement('div');
            resultCard.className = `model-result ${isBestModel ? 'best-model' : ''}`;

            // Create the card content with improved digit display
            resultCard.innerHTML = `
                ${isBestModel ? '<div class="best-model-badge">★</div>' : ''}
                <div class="model-result-name">${modelInfo.name}</div>
                <div class="model-result-digit">${prediction.digit}</div>
                <div class="model-result-confidence">
                    <span class="confidence-value-large">${confidence}%</span> confidence
                </div>
                <div class="model-result-detail">
                    <div class="confidence-label-row">
                        <span>Probability Distribution</span>
                        <span>Digit ${prediction.digit}</span>
                    </div>
                    <div class="confidence-small-bar">
                        <div class="confidence-small-bar-fill" style="width: ${confidence}%; background-color: var(--primary-color);"></div>
                    </div>
                    <div class="confidence-label-row">
                        <span>Click for details</span>
                        <span>${confidence}%</span>
                    </div>
                </div>
            `;

            // Add the card to the container
            modelResultsContainer.appendChild(resultCard);

            // Add click event to expand the details
            resultCard.addEventListener('click', () => {
                this.expandModelDetails(modelType, prediction);
            });
        }

        // Scroll to the comparison section
        setTimeout(() => {
            comparisonSection.scrollIntoView({ behavior: 'smooth' });
        }, 300);
    }

    expandModelDetails(modelType, prediction) {
        // Create a modal for detailed view
        const modal = document.createElement('div');
        modal.className = 'model-detail-modal';

        const modelInfo = this.models[modelType];
        const confidence = prediction.confidences[prediction.digit].toFixed(1);

        modal.innerHTML = `
            <div class="model-detail-content">
                <div class="modal-header">
                    <h3>${modelInfo.name} - Detail View</h3>
                    <button class="close-detail-modal">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path d="M12 10.586l4.95-4.95 1.414 1.414-4.95 4.95 4.95 4.95-1.414 1.414-4.95-4.95-4.95 4.95-1.414-1.414 4.95-4.95-4.95-4.95L7.05 5.636z"/></svg>
                    </button>
                </div>
                <div class="model-detail-body">
                    <div class="model-detail-header">
                        <div class="detail-digit">${prediction.digit}</div>
                        <div class="detail-info">
                            <p class="detail-confidence">Confidence: <span class="highlight">${confidence}%</span></p>
                            <p class="detail-desc">${modelInfo.description}</p>
                        </div>
                    </div>
                    <div class="confidence-distribution-detail">
                        <h4>Probability Distribution</h4>
                        <div class="detail-bars-container"></div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Generate detailed confidence bars
        const barsContainer = modal.querySelector('.detail-bars-container');

        for (let i = 0; i < 10; i++) {
            const conf = prediction.confidences[i];
            const isPredicted = i === prediction.digit;

            const barElement = document.createElement('div');
            barElement.className = 'detail-confidence-bar';

            barElement.innerHTML = `
                <div class="detail-digit-label">${i}</div>
                <div class="detail-bar-container">
                    <div class="detail-bar-fill ${isPredicted ? 'predicted' : ''}" style="width: 0%"></div>
                </div>
                <div class="detail-confidence-value">${conf.toFixed(1)}%</div>
            `;

            barsContainer.appendChild(barElement);
        }

        // Add close event
        const closeButton = modal.querySelector('.close-detail-modal');
        closeButton.addEventListener('click', () => {
            modal.classList.add('hiding');
            setTimeout(() => {
                document.body.removeChild(modal);
            }, 300);
        });

        // Add style for modal if not already added
        this.addModalStyles();

        // Animate the modal in
        setTimeout(() => {
            modal.classList.add('visible');
        }, 10);

        // Animate the bars after a short delay
        setTimeout(() => {
            const bars = modal.querySelectorAll('.detail-bar-fill');
            bars.forEach((bar, index) => {
                setTimeout(() => {
                    bar.style.width = `${prediction.confidences[index]}%`;
                }, index * 50);
            });
        }, 300);
    }

    addModalStyles() {
        // Only add styles once
        if (document.getElementById('model-detail-modal-styles')) return;

        const style = document.createElement('style');
        style.id = 'model-detail-modal-styles';
        style.textContent = `
            .model-detail-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.3s ease;
            }
            
            .model-detail-modal.visible {
                opacity: 1;
                visibility: visible;
            }
            
            .model-detail-modal.hiding {
                opacity: 0;
            }
            
            .model-detail-content {
                background-color: var(--surface-color);
                border-radius: var(--radius-lg);
                width: 90%;
                max-width: 600px;
                max-height: 90vh;
                overflow-y: auto;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                transform: scale(0.9);
                transition: transform 0.3s ease;
            }
            
            .model-detail-modal.visible .model-detail-content {
                transform: scale(1);
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: var(--spacing-lg);
                border-bottom: 1px solid var(--border-color);
            }
            
            .modal-header h3 {
                margin: 0;
                color: var(--primary-color);
            }
            
            .close-detail-modal {
                background: none;
                border: none;
                cursor: pointer;
                display: flex;
                padding: 0;
            }
            
            .close-detail-modal svg {
                fill: var(--text-secondary);
                width: 24px;
                height: 24px;
                transition: fill 0.2s ease;
            }
            
            .close-detail-modal:hover svg {
                fill: var(--text-color);
            }
            
            .model-detail-body {
                padding: var(--spacing-lg);
            }
            
            .model-detail-header {
                display: flex;
                margin-bottom: var(--spacing-xl);
            }
            
            .detail-digit {
                font-size: 4rem;
                font-weight: 700;
                width: 100px;
                height: 100px;
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: var(--primary-color);
                color: white;
                border-radius: var(--radius-md);
                margin-right: var(--spacing-lg);
                box-shadow: 0 6px 15px rgba(79, 70, 229, 0.3);
            }
            
            .detail-info {
                flex: 1;
            }
            
            .detail-confidence {
                font-size: 1.2rem;
                margin-bottom: var(--spacing-sm);
            }
            
            .highlight {
                color: var(--primary-color);
                font-weight: 700;
            }
            
            .detail-desc {
                color: var(--text-secondary);
                margin-bottom: var(--spacing-md);
            }
            
            .confidence-distribution-detail h4 {
                margin-bottom: var(--spacing-md);
            }
            
            .detail-confidence-bar {
                display: flex;
                align-items: center;
                height: 40px;
                margin-bottom: var(--spacing-sm);
            }
            
            .detail-digit-label {
                width: 30px;
                font-weight: 600;
                font-size: 1.1rem;
            }
            
            .detail-bar-container {
                flex: 1;
                height: 16px;
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                overflow: hidden;
                margin: 0 var(--spacing-md);
            }
            
            .dark-theme .detail-bar-container {
                background-color: rgba(255, 255, 255, 0.1);
            }
            
            .detail-bar-fill {
                height: 100%;
                width: 0;
                background-color: var(--secondary-color);
                border-radius: 8px;
                transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
            }
            
            .detail-bar-fill.predicted {
                background-color: var(--primary-color);
            }
            
            .detail-confidence-value {
                font-size: 0.95rem;
                min-width: 60px;
                text-align: right;
                font-variant-numeric: tabular-nums;
                font-weight: 600;
            }
            
            @media (max-width: 768px) {
                .model-detail-header {
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                }
                
                .detail-digit {
                    margin-right: 0;
                    margin-bottom: var(--spacing-md);
                }
            }
        `;

        document.head.appendChild(style);
    }

    handleRecognitionError(mode) {
        // Determine which elements to use based on mode
        const prefix = mode === 'comparison' ? 'comparison-' : 'single-';

        // Hide loading spinner
        document.getElementById(`${prefix}loading-spinner`).classList.add('hidden');

        // Show an error message to the user
        document.getElementById(`${prefix}waiting-message`).classList.remove('hidden');
        document.getElementById(`${prefix}waiting-message`).innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64"><path fill="none" d="M0 0h24v24H0z"/><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-7v2h2v-2h-2zm0-8v6h2V7h-2z"/></svg>
            <p>Error recognizing digit. Please try again.</p>
        `;

        // Reset the message after a delay
        setTimeout(() => {
            let message = '';
            if (mode === 'comparison') {
                message = `
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64"><path fill="none" d="M0 0h24v24H0z"/><path d="M5.463 4.433A9.961 9.961 0 0 1 12 2c5.523 0 10 4.477 10 10 0 2.136-.67 4.116-1.81 5.74L17 12h3A8 8 0 0 0 6.46 6.228l-.997-1.795zm13.074 15.134A9.961 9.961 0 0 1 12 22C6.477 22 2 17.523 2 12c0-2.136.67-4.116 1.81-5.74L7 12H4a8 8 0 0 0 13.54 5.772l.997 1.795z"/></svg>
                    <p>Draw a digit and click "Compare All Models" to see the results</p>
                `;
            } else {
                message = `
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64"><path fill="none" d="M0 0h24v24H0z"/><path d="M5.463 4.433A9.961 9.961 0 0 1 12 2c5.523 0 10 4.477 10 10 0 2.136-.67 4.116-1.81 5.74L17 12h3A8 8 0 0 0 6.46 6.228l-.997-1.795zm13.074 15.134A9.961 9.961 0 0 1 12 22C6.477 22 2 17.523 2 12c0-2.136.67-4.116 1.81-5.74L7 12H4a8 8 0 0 0 13.54 5.772l.997 1.795z"/></svg>
                    <p>Draw a digit and click "Recognize" to see the results</p>
                `;
            }

            document.getElementById(`${prefix}waiting-message`).innerHTML = message;
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