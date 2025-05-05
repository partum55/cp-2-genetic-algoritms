/**
 * Simplified Canvas functionality for digit drawing
 * Focuses only on drawing lines without any extra effects
 */
class DrawingCanvas {
    constructor() {
        this.canvas = document.getElementById('drawing-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvasOverlay = document.querySelector('.canvas-overlay');
        this.isDrawing = false;
        this.hasDrawn = false;

        // Configure canvas
        this.setupCanvas();

        // Add event listeners
        this.addEventListeners();

        // Clear canvas initially
        this.clearCanvas();
    }

    setupCanvas() {
        // Set canvas size
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;

        // Set drawing properties
        this.ctx.lineWidth = 18;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = 'black';
    }

    addEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));

        // Clear button
        const clearButton = document.getElementById('clear-button');
        clearButton.addEventListener('click', this.clearCanvas.bind(this));

        // Handle window resize
        window.addEventListener('resize', this.handleResize.bind(this));

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyPress.bind(this));
    }

    startDrawing(e) {
        this.isDrawing = true;
        const coords = this.getCoordinates(e);

        // Start a new path
        this.ctx.beginPath();
        this.ctx.moveTo(coords.x, coords.y);

        // Hide the canvas overlay when drawing starts
        if (!this.hasDrawn) {
            if (this.canvasOverlay) {
                this.canvasOverlay.classList.add('hidden');
            }
            this.hasDrawn = true;
        }

        // Prevent default behavior
        e.preventDefault();
    }

    draw(e) {
        if (!this.isDrawing) return;

        const coords = this.getCoordinates(e);

        // Draw line to the new position
        this.ctx.lineTo(coords.x, coords.y);
        this.ctx.stroke();

        // Prevent default behavior
        e.preventDefault();
    }

    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
        }
    }

    clearCanvas() {
        // Fill with white background
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Show the canvas overlay
        if (this.canvasOverlay) {
            this.canvasOverlay.classList.remove('hidden');
        }
        this.hasDrawn = false;

        // Hide any prediction results
        const predictionResult = document.getElementById('prediction-result');
        const waitingMessage = document.getElementById('waiting-message');
        const loadingSpinner = document.getElementById('loading-spinner');
        const comparisonSection = document.getElementById('comparison-section');

        if (predictionResult) predictionResult.classList.add('hidden');
        if (waitingMessage) waitingMessage.classList.remove('hidden');
        if (loadingSpinner) loadingSpinner.classList.add('hidden');
        if (comparisonSection && !comparisonSection.classList.contains('hidden')) {
            comparisonSection.classList.add('hidden');
        }
    }

    handleTouchStart(e) {
        // Prevent default to avoid scrolling/zooming
        e.preventDefault();

        if (e.touches.length === 1) {
            const touch = e.touches[0];
            const rect = this.canvas.getBoundingClientRect();
            const offsetX = touch.clientX - rect.left;
            const offsetY = touch.clientY - rect.top;

            // Create a synthetic mouse event
            const mouseEvent = {
                clientX: touch.clientX,
                clientY: touch.clientY,
                preventDefault: () => {}
            };

            this.startDrawing(mouseEvent);
        }
    }

    handleTouchMove(e) {
        // Prevent default to avoid scrolling/zooming
        e.preventDefault();

        if (e.touches.length === 1 && this.isDrawing) {
            const touch = e.touches[0];

            // Create a synthetic mouse event
            const mouseEvent = {
                clientX: touch.clientX,
                clientY: touch.clientY,
                preventDefault: () => {}
            };

            this.draw(mouseEvent);
        }
    }

    handleResize() {
        // Save the current drawing
        const imageData = this.canvas.toDataURL('image/png');

        // Resize canvas
        this.setupCanvas();

        // Restore drawing
        const img = new Image();
        img.onload = () => {
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
        };
        img.src = imageData;
    }

    handleKeyPress(e) {
        // 'C' key for clear
        if (e.key === 'c' || e.key === 'C') {
            this.clearCanvas();
        }
    }

    getCoordinates(event) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }

    getImageData() {
        // Return the image as a data URL
        return this.canvas.toDataURL('image/png');
    }
}

// Initialize canvas when window loads
window.addEventListener('load', () => {
    window.drawingCanvas = new DrawingCanvas();
});