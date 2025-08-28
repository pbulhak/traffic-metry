/**
 * TrafficMetry WebSocket Client
 * Manages WebSocket connection with auto-reconnect functionality
 */

class TrafficMetryWebSocket {
    constructor(url, options = {}) {
        this.url = url;
        this.options = {
            reconnectInterval: 1000,       // Start with 1s
            maxReconnectInterval: 30000,   // Max 30s
            reconnectDecay: 1.5,          // Exponential backoff multiplier
            maxReconnectAttempts: 50,     // Max reconnection attempts
            timeoutInterval: 2000,        // Connection timeout
            ...options
        };
        
        // Connection state
        this.ws = null;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.isManualClose = false;
        this.connectionStart = null;
        
        // Event listeners registry
        this.listeners = {
            connection_status: [],
            vehicle_event: [],
            error: [],
            raw_message: []
        };
        
        // Connection states
        this.states = {
            CONNECTING: 'connecting',
            CONNECTED: 'connected', 
            ERROR: 'error',
            CLOSED: 'closed'
        };
        
        this.currentState = this.states.CONNECTING;
    }
    
    /**
     * Initialize WebSocket connection
     */
    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
            console.log('WebSocket already connecting or connected');
            return;
        }
        
        this.isManualClose = false;
        this.connectionStart = Date.now();
        this._updateConnectionStatus(this.states.CONNECTING);
        
        console.log(`Attempting to connect to WebSocket: ${this.url}`);
        
        try {
            this.ws = new WebSocket(this.url);
            this._setupEventHandlers();
            
            // Connection timeout
            setTimeout(() => {
                if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
                    console.log('WebSocket connection timeout');
                    this.ws.close();
                    this._handleConnectionTimeout();
                }
            }, this.options.timeoutInterval);
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this._handleError(error);
        }
    }
    
    /**
     * Manually disconnect WebSocket
     */
    disconnect() {
        this.isManualClose = true;
        this._clearReconnectTimer();
        
        if (this.ws) {
            this.ws.close(1000, 'Manual disconnect');
        }
        
        this._updateConnectionStatus(this.states.CLOSED);
        console.log('WebSocket manually disconnected');
    }
    
    /**
     * Send message to WebSocket server
     */
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                const messageStr = typeof message === 'string' ? message : JSON.stringify(message);
                this.ws.send(messageStr);
                console.log('Message sent:', message);
                return true;
            } catch (error) {
                console.error('Error sending message:', error);
                this.emit('error', { type: 'send_error', error });
                return false;
            }
        } else {
            console.warn('WebSocket not connected. Cannot send message:', message);
            return false;
        }
    }
    
    /**
     * Register event listener
     */
    on(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event].push(callback);
        } else {
            console.warn(`Unknown event type: ${event}`);
        }
    }
    
    /**
     * Remove event listener
     */
    off(event, callback) {
        if (this.listeners[event]) {
            const index = this.listeners[event].indexOf(callback);
            if (index > -1) {
                this.listeners[event].splice(index, 1);
            }
        }
    }
    
    /**
     * Emit event to registered listeners
     */
    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }
    
    /**
     * Get current connection status
     */
    getConnectionStatus() {
        return {
            state: this.currentState,
            reconnectAttempts: this.reconnectAttempts,
            connectionTime: this.connectionStart ? Date.now() - this.connectionStart : 0,
            url: this.url
        };
    }
    
    // Private Methods
    
    /**
     * Setup WebSocket event handlers
     */
    _setupEventHandlers() {
        this.ws.onopen = (event) => this._handleOpen(event);
        this.ws.onmessage = (event) => this._handleMessage(event);
        this.ws.onclose = (event) => this._handleClose(event);
        this.ws.onerror = (error) => this._handleError(error);
    }
    
    /**
     * Handle WebSocket connection opened
     */
    _handleOpen(event) {
        console.log('WebSocket connected successfully');
        this._updateConnectionStatus(this.states.CONNECTED);
        this._resetReconnectAttempts();
        this.emit('connection_status', {
            status: this.states.CONNECTED,
            event: event,
            connectionTime: Date.now() - this.connectionStart
        });
    }
    
    /**
     * Handle incoming WebSocket messages
     */
    _handleMessage(event) {
        try {
            this.emit('raw_message', event.data);
            const eventData = JSON.parse(event.data);
            console.log('WebSocket message received:', eventData);

            // Zgodnie z API v2.3, cały obiekt to event.
            // Sprawdzamy, czy ma kluczowe pole, np. eventId.
            if (eventData.eventId && eventData.vehicleType) {
                this.emit('vehicle_event', eventData);
            } else {
                console.warn('Received unknown message format:', eventData);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error, 'Raw data:', event.data);
            this.emit('error', { type: 'parse_error', error, rawData: event.data });
        }
    }
    
    /**
     * Handle WebSocket connection closed
     */
    _handleClose(event) {
        console.log('WebSocket connection closed:', event.code, event.reason);
        
        this._updateConnectionStatus(this.states.CLOSED);
        
        this.emit('connection_status', {
            status: this.states.CLOSED,
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
        });
        
        // Auto-reconnect if not manual close and within attempt limits
        if (!this.isManualClose && this.reconnectAttempts < this.options.maxReconnectAttempts) {
            this._scheduleReconnect();
        } else if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached. Giving up.');
            this.emit('error', {
                type: 'max_reconnect_attempts',
                attempts: this.reconnectAttempts
            });
        }
    }
    
    /**
     * Handle WebSocket errors
     */
    _handleError(error) {
        console.error('WebSocket error:', error);
        this._updateConnectionStatus(this.states.ERROR);
        
        this.emit('error', {
            type: 'websocket_error',
            error: error
        });
        
        this.emit('connection_status', {
            status: this.states.ERROR,
            error: error
        });
    }
    
    /**
     * Handle connection timeout
     */
    _handleConnectionTimeout() {
        console.error('WebSocket connection timeout');
        this._updateConnectionStatus(this.states.ERROR);
        
        this.emit('error', {
            type: 'connection_timeout',
            timeout: this.options.timeoutInterval
        });
        
        if (!this.isManualClose) {
            this._scheduleReconnect();
        }
    }
    
    /**
     * Schedule reconnection attempt with exponential backoff
     */
    _scheduleReconnect() {
        this._clearReconnectTimer();
        
        this.reconnectAttempts++;
        
        // Calculate reconnect interval with exponential backoff
        const interval = Math.min(
            this.options.reconnectInterval * Math.pow(this.options.reconnectDecay, this.reconnectAttempts - 1),
            this.options.maxReconnectInterval
        );
        
        console.log(`Scheduling reconnect attempt ${this.reconnectAttempts}/${this.options.maxReconnectAttempts} in ${interval}ms`);
        
        this.reconnectTimer = setTimeout(() => {
            console.log(`Reconnect attempt ${this.reconnectAttempts}`);
            this.connect();
        }, interval);
    }
    
    /**
     * Clear reconnection timer
     */
    _clearReconnectTimer() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }
    
    /**
     * Reset reconnection attempts counter
     */
    _resetReconnectAttempts() {
        this.reconnectAttempts = 0;
        this._clearReconnectTimer();
    }
    
    /**
     * Update current connection status
     */
    _updateConnectionStatus(newStatus) {
        if (this.currentState !== newStatus) {
            const previousState = this.currentState;
            this.currentState = newStatus;
            console.log(`Connection status changed: ${previousState} -> ${newStatus}`);
        }
    }
}