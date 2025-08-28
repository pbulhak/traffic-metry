/**
 * TrafficMetry Main Application
 * Step 3: WebSocket connection and console logging only
 * Visualization will be implemented in Step 4
 */

class TrafficMetryApp {
    constructor() {
        // WebSocket client
        this.wsClient = null;
        this.wsUrl = 'ws://localhost:8000/ws';
        
        // DOM elements
        this.connectionStatus = null;
        this.statusIndicator = null;
        this.statusText = null;
        this.eventsCount = null;
        this.connectionTime = null;
        this.placeholderMessage = null;
        
        // App state
        this.eventsReceived = 0;
        this.connectionStartTime = null;
        this.connectionTimer = null;
        
        // Bind methods
        this.handleVehicleEvent = this.handleVehicleEvent.bind(this);
        this.handleConnectionStatus = this.handleConnectionStatus.bind(this);
        this.handleError = this.handleError.bind(this);
        this.updateConnectionTimer = this.updateConnectionTimer.bind(this);
    }
    
    /**
     * Initialize the application
     */
    init() {
        console.log('TrafficMetry App initializing...');
        document.addEventListener('DOMContentLoaded', () => {
            this._initializeApp();
        });
    }
    
    /**
     * Initialize app after DOM is ready
     */
    _initializeApp() {
        console.log('DOM ready, initializing app components...');
        
        // Get DOM elements
        this._initializeDOMElements();
        
        // Setup WebSocket client
        this._initializeWebSocket();
        
        // Setup event listeners
        this._setupEventListeners();
        
        console.log('TrafficMetry App initialized successfully');
    }
    
    /**
     * Initialize DOM element references
     */
    _initializeDOMElements() {
        this.connectionStatus = document.getElementById('connectionStatus');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.eventsCount = document.getElementById('eventsCount');
        this.connectionTime = document.getElementById('connectionTime');
        this.placeholderMessage = document.getElementById('placeholderMessage');
        
        // Verify all elements are found
        const elements = {
            connectionStatus: this.connectionStatus,
            statusIndicator: this.statusIndicator,
            statusText: this.statusText,
            eventsCount: this.eventsCount,
            connectionTime: this.connectionTime,
            placeholderMessage: this.placeholderMessage
        };
        
        for (const [name, element] of Object.entries(elements)) {
            if (!element) {
                console.error(`Required DOM element not found: ${name}`);
            }
        }
    }
    
    /**
     * Initialize WebSocket client and connect
     */
    _initializeWebSocket() {
        console.log(`Creating WebSocket client for: ${this.wsUrl}`);
        
        // Create WebSocket client with custom options
        this.wsClient = new TrafficMetryWebSocket(this.wsUrl, {
            reconnectInterval: 1000,
            maxReconnectInterval: 15000,
            reconnectDecay: 1.3,
            maxReconnectAttempts: 30
        });
        
        // Register event listeners
        this.wsClient.on('vehicle_event', this.handleVehicleEvent);
        this.wsClient.on('connection_status', this.handleConnectionStatus);
        this.wsClient.on('error', this.handleError);
        
        // Start connection
        this.wsClient.connect();
    }
    
    /**
     * Setup additional event listeners
     */
    _setupEventListeners() {
        // Handle page visibility change (reconnect on focus)
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.wsClient) {
                const status = this.wsClient.getConnectionStatus();
                if (status.state !== 'connected') {
                    console.log('Page became visible, checking connection...');
                    this.wsClient.connect();
                }
            }
        });
        
        // Handle beforeunload (cleanup)
        window.addEventListener('beforeunload', () => {
            if (this.wsClient) {
                this.wsClient.disconnect();
            }
            this._clearConnectionTimer();
        });
    }
    
    /**
     * Handle incoming vehicle events
     * Step 3: Only console logging, no visualization yet
     */
    handleVehicleEvent(eventData) {
        const {
            eventId = 'N/A',
            timestamp = new Date().toISOString(),
            vehicleType = 'unknown',
            vehicleId = 'N/A',
            movement = { lane: 'N/A', direction: 'N/A' },
            analytics = { confidence: 0 },
            position = { boundingBox: {} }
        } = eventData;

        console.group(`🚗 Vehicle Event Received: ${vehicleType}`);
        console.log('Event ID:', eventId);
        console.log('Timestamp:', timestamp);
        console.log('Vehicle ID:', vehicleId);
        console.log('Lane:', movement.lane);
        console.log('Direction:', movement.direction);
        console.log('Confidence:', analytics.confidence);
        console.log('Full Event Data:', eventData);
        console.groupEnd();

        this.eventsReceived++;
        this._updateEventsCounter();

        if (this.placeholderMessage && this.placeholderMessage.style.display !== 'none') {
            this.placeholderMessage.style.display = 'none';
        }
    }
    
    /**
     * Handle WebSocket connection status changes
     */
    handleConnectionStatus(statusData) {
        console.log('Connection Status Update:', statusData);
        
        const status = statusData.status;
        
        // Update UI based on status
        this._updateConnectionStatusUI(status, statusData);
        
        // Handle connection timing
        if (status === 'connected') {
            this.connectionStartTime = Date.now();
            this._startConnectionTimer();
        } else if (status === 'closed' || status === 'error') {
            this._clearConnectionTimer();
        }
    }
    
    /**
     * Handle WebSocket errors
     */
    handleError(errorData) {
        console.error('WebSocket Error:', errorData);
        
        // Log different error types
        switch (errorData.type) {
            case 'websocket_error':
                console.error('WebSocket connection error:', errorData.error);
                break;
            case 'connection_timeout':
                console.error('Connection timeout after', errorData.timeout, 'ms');
                break;
            case 'max_reconnect_attempts':
                console.error('Maximum reconnection attempts reached:', errorData.attempts);
                break;
            case 'parse_error':
                console.error('Message parsing error:', errorData.error, 'Raw data:', errorData.rawData);
                break;
            default:
                console.error('Unknown error type:', errorData.type, errorData);
        }
    }
    
    /**
     * Update connection status UI elements
     */
    _updateConnectionStatusUI(status, statusData = {}) {
        if (!this.statusIndicator || !this.statusText) {
            return;
        }
        
        // Clear previous status classes
        this.statusIndicator.className = 'status-indicator';
        
        // Update status indicator and text
        switch (status) {
            case 'connecting':
                this.statusIndicator.classList.add('connecting');
                this.statusText.textContent = 'Connecting...';
                break;
                
            case 'connected':
                this.statusIndicator.classList.add('connected');
                this.statusText.textContent = 'Connected';
                break;
                
            case 'error':
                this.statusIndicator.classList.add('error');
                this.statusText.textContent = 'Connection Error';
                break;
                
            case 'closed':
                this.statusIndicator.classList.add('disconnected');
                if (statusData.code === 1000) {
                    this.statusText.textContent = 'Disconnected';
                } else {
                    this.statusText.textContent = 'Connection Lost';
                }
                break;
                
            default:
                this.statusIndicator.classList.add('disconnected');
                this.statusText.textContent = 'Unknown Status';
        }
        
        console.log(`UI status updated: ${status}`);
    }
    
    /**
     * Update events counter in UI
     */
    _updateEventsCounter() {
        if (this.eventsCount) {
            this.eventsCount.textContent = this.eventsReceived.toString();
        }
    }
    
    /**
     * Start connection timer to show connection duration
     */
    _startConnectionTimer() {
        this._clearConnectionTimer();
        
        this.connectionTimer = setInterval(this.updateConnectionTimer, 1000);
    }
    
    /**
     * Update connection timer display
     */
    updateConnectionTimer() {
        if (!this.connectionTime || !this.connectionStartTime) {
            return;
        }
        
        const elapsed = Math.floor((Date.now() - this.connectionStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        
        this.connectionTime.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    /**
     * Clear connection timer
     */
    _clearConnectionTimer() {
        if (this.connectionTimer) {
            clearInterval(this.connectionTimer);
            this.connectionTimer = null;
        }
        
        if (this.connectionTime) {
            this.connectionTime.textContent = '--:--';
        }
    }
}

// Initialize app when script loads
const app = new TrafficMetryApp();
app.init();