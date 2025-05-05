/**
 * Updated application script with mode selection functionality
 */
class MnistApp {
    constructor() {
        // References to DOM elements
        this.modeSelectionContainer = document.getElementById('mode-selection-container');
        this.singleModelContainer = document.getElementById('single-model-container');
        this.comparisonContainer = document.getElementById('comparison-container');

        this.singleModelMode = document.getElementById('single-model-mode');
        this.compareModelsMode = document.getElementById('compare-models-mode');

        this.singleReturnButton = document.getElementById('single-return-button');
        this.comparisonReturnButton = document.getElementById('comparison-return-button');

        this.themeSwitch = document.getElementById('theme-switch');
        this.showShortcutsButton = document.getElementById('show-shortcuts');
        this.shortcutsModal = document.getElementById('shortcuts-modal');
        this.closeModalButton = document.querySelector('.close-modal');

        // Track current mode
        this.currentMode = 'selection'; // 'selection', 'single', or 'comparison'

        // Initialize app elements
        this.initializeApp();
    }

    initializeApp() {
        // Set up mode selection buttons
        this.singleModelMode.addEventListener('click', () => this.setMode('single'));
        this.compareModelsMode.addEventListener('click', () => this.setMode('comparison'));

        // Set up return buttons
        this.singleReturnButton.addEventListener('click', () => this.setMode('selection'));
        this.comparisonReturnButton.addEventListener('click', () => this.setMode('selection'));

        // Set up theme toggle
        this.themeSwitch.addEventListener('change', this.toggleTheme.bind(this));

        // Set up theme based on user preference
        this.initializeTheme();

        // Set up shortcuts modal
        this.showShortcutsButton.addEventListener('click', this.showShortcuts.bind(this));
        this.closeModalButton.addEventListener('click', this.hideShortcuts.bind(this));

        // Close modal when clicking outside of it
        this.shortcutsModal.addEventListener('click', (e) => {
            if (e.target === this.shortcutsModal) {
                this.hideShortcuts();
            }
        });

        // Add keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyEvents.bind(this));

        // Check if backend is available
        this.checkBackendConnection();

        // Add animation effects
        this.addScrollAnimations();
    }

    setMode(mode) {
        this.currentMode = mode;

        // Hide all containers first
        this.modeSelectionContainer.classList.add('hidden');
        this.singleModelContainer.classList.add('hidden');
        this.comparisonContainer.classList.add('hidden');

        // Show the appropriate container
        if (mode === 'selection') {
            this.modeSelectionContainer.classList.remove('hidden');
        } else if (mode === 'single') {
            this.singleModelContainer.classList.remove('hidden');

            // Initialize single model canvas if needed
            if (!window.singleDrawingCanvas) {
                window.singleDrawingCanvas = new DrawingCanvas('single-drawing-canvas', {
                    clearButtonId: 'single-clear-button',
                    waitingMessageId: 'single-waiting-message',
                    loadingSpinnerId: 'single-loading-spinner',
                    predictionResultId: 'single-prediction-result'
                });
            }

            // Make sure model manager is in single mode
            if (window.modelManager) {
                window.modelManager.setComparisonMode(false);
            }
        } else if (mode === 'comparison') {
            this.comparisonContainer.classList.remove('hidden');

            // Initialize comparison canvas if needed
            if (!window.comparisonDrawingCanvas) {
                window.comparisonDrawingCanvas = new DrawingCanvas('comparison-drawing-canvas', {
                    clearButtonId: 'comparison-clear-button',
                    waitingMessageId: 'comparison-waiting-message',
                    loadingSpinnerId: 'comparison-loading-spinner',
                    predictionResultId: 'comparison-prediction-result'
                });
            }

            // Make sure model manager is in comparison mode
            if (window.modelManager) {
                window.modelManager.setComparisonMode(true);
            }

            // Hide comparison results initially
            const comparisonResultsSection = document.getElementById('comparison-results-section');
            if (comparisonResultsSection) {
                comparisonResultsSection.classList.add('hidden');
            }
        }

        // Scroll to top when changing modes
        window.scrollTo(0, 0);
    }

    toggleTheme(e) {
        const isDarkTheme = e.target.checked;
        document.body.classList.toggle('dark-theme', isDarkTheme);

        // Save preference to localStorage
        localStorage.setItem('darkTheme', isDarkTheme);
    }

    initializeTheme() {
        // Check for saved theme preference
        const savedTheme = localStorage.getItem('darkTheme');
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

        // Apply dark theme if explicitly saved as dark or if user's system prefers dark
        // and there's no saved preference
        const shouldUseDarkTheme = savedTheme === 'true' || (savedTheme === null && prefersDark);

        document.body.classList.toggle('dark-theme', shouldUseDarkTheme);
        this.themeSwitch.checked = shouldUseDarkTheme;

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (localStorage.getItem('darkTheme') === null) {
                // Only auto-switch if the user hasn't explicitly chosen a theme
                const shouldUseDarkTheme = e.matches;
                document.body.classList.toggle('dark-theme', shouldUseDarkTheme);
                this.themeSwitch.checked = shouldUseDarkTheme;
            }
        });
    }

    showShortcuts() {
        this.shortcutsModal.classList.add('visible');
        this.shortcutsModal.classList.remove('hidden');
    }

    hideShortcuts() {
        this.shortcutsModal.classList.remove('visible');
        // Use a timeout to allow the fade-out animation to complete
        setTimeout(() => {
            this.shortcutsModal.classList.add('hidden');
        }, 300);
    }

    handleKeyEvents(e) {
        // Theme toggle with 'T' key
        if (e.key === 't' || e.key === 'T') {
            this.themeSwitch.checked = !this.themeSwitch.checked;
            this.toggleTheme({ target: this.themeSwitch });
        }

        // Show shortcuts with '?' key
        if (e.key === '?') {
            this.showShortcuts();
        }

        // Close shortcuts modal with 'Escape' key
        if (e.key === 'Escape' && !this.shortcutsModal.classList.contains('hidden')) {
            this.hideShortcuts();
        }

        // Add mode selection shortcuts
        if (this.currentMode === 'selection') {
            if (e.key === '1') {
                this.setMode('single');
            } else if (e.key === '2') {
                this.setMode('comparison');
            }
        }

        // Escape key to return to mode selection
        if ((this.currentMode === 'single' || this.currentMode === 'comparison') && e.key === 'Escape') {
            this.setMode('selection');
        }
    }

    async checkBackendConnection() {
        try {
            const response = await fetch('http://localhost:5000/api/health', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Set a short timeout to avoid waiting too long
                signal: AbortSignal.timeout(3000),
            });

            if (response.ok) {
                console.log('Backend connection successful');
                const data = await response.json();
                console.log('Available models:', data.models);
                console.log('Device:', data.device);

                // Show a subtle success notification
                this.showConnectionStatus(true);
            } else {
                this.showBackendWarning();
                this.showConnectionStatus(false);
            }
        } catch (error) {
            console.warn('Backend not available:', error);
            this.showBackendWarning();
            this.showConnectionStatus(false);
        }
    }

    showConnectionStatus(isConnected) {
        // Create or get the connection status element
        let statusElement = document.querySelector('.connection-status');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.className = 'connection-status';
            document.body.appendChild(statusElement);

            // Add styles
            const style = document.createElement('style');
            style.textContent = `
                .connection-status {
                    position: fixed;
                    bottom: 10px;
                    right: 10px;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                    z-index: 50;
                }
                .connection-status.connected {
                    background-color: rgba(16, 185, 129, 0.1);
                    color: #10b981;
                    border: 1px solid #10b981;
                }
                .connection-status.disconnected {
                    background-color: rgba(239, 68, 68, 0.1);
                    color: #ef4444;
                    border: 1px solid #ef4444;
                }
                .connection-status.visible {
                    opacity: 1;
                }
            `;
            document.head.appendChild(style);
        }

        // Update status
        if (isConnected) {
            statusElement.textContent = 'Connected to backend';
            statusElement.classList.add('connected');
            statusElement.classList.remove('disconnected');
        } else {
            statusElement.textContent = 'Running in demo mode';
            statusElement.classList.add('disconnected');
            statusElement.classList.remove('connected');
        }

        // Show the status
        statusElement.classList.add('visible');

        // Hide after 3 seconds
        setTimeout(() => {
            statusElement.classList.remove('visible');
        }, 3000);
    }

    showBackendWarning() {
        // Create a warning message that the backend is not available
        const warning = document.createElement('div');
        warning.className = 'backend-warning';
        warning.innerHTML = `
            <div class="warning-content">
                <h3>Backend Server Unavailable</h3>
                <p>The application will run in demo mode with simulated predictions.</p>
                <p>To use real models, make sure the Flask backend server is running:</p>
                <div class="code-block">
                    <code>python app.py</code>
                </div>
                <button id="dismiss-warning" class="btn primary">Got it</button>
            </div>
        `;
        document.body.appendChild(warning);

        // Add dismiss functionality
        document.getElementById('dismiss-warning').addEventListener('click', () => {
            warning.style.opacity = '0';
            setTimeout(() => {
                warning.remove();
            }, 500);
        });
    }

    addScrollAnimations() {
        // Add scroll-triggered animations to elements
        const animateOnScroll = (entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate');
                    observer.unobserve(entry.target);
                }
            });
        };

        const observer = new IntersectionObserver(animateOnScroll, {
            root: null,
            threshold: 0.1,
            rootMargin: '-20px'
        });

        // Apply to elements you want to animate
        document.querySelectorAll('.model-selection, .main-content, .footer, .model-results').forEach(el => {
            el.classList.add('scroll-animate');
            observer.observe(el);
        });
    }
}

// Initialize the app when document is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mnistApp = new MnistApp();
});