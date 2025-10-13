/**
 * VehicleRenderer - SVG Sprite Visualization Engine
 * Manages dynamic creation, positioning and animation of vehicle sprites
 */

class VehicleRenderer {
    constructor(containerSelector = '.traffic-visualization') {
        this.container = document.querySelector(containerSelector);
        this.activeVehicles = new Map(); // vehicleId -> DOM element
        this.vehicleCleanupData = new Map(); // vehicleId -> {listener, timeout}
        this.animationDuration = 3000; // 3 seconds
        this.maxVehicles = 20; // Performance limit
        
        if (!this.container) {
            console.error('VehicleRenderer: Container not found:', containerSelector);
            return;
        }
        
        console.log('VehicleRenderer initialized with container:', containerSelector);
    }
    
    /**
     * Create and animate a vehicle sprite based on event data
     */
    createVehicle(eventData) {
        const {
            vehicleId = 'unknown',
            vehicleType = 'other_vehicle',
            movement = {}
        } = eventData;

        const direction = movement.direction || 'right';

        // Prevent duplicate vehicles
        if (this.activeVehicles.has(vehicleId)) {
            return;
        }

        // Performance limit check
        if (this.activeVehicles.size >= this.maxVehicles) {
            console.warn('Max vehicles limit reached, skipping new vehicle');
            return;
        }

        try {
            // Create vehicle element
            const vehicleElement = this._createVehicleElement(vehicleId, vehicleType, direction);

            // Add to container and tracking
            this.container.appendChild(vehicleElement);
            this.activeVehicles.set(vehicleId, vehicleElement);

        } catch (error) {
            console.error('Error creating vehicle:', error, eventData);
        }
    }
    
    /**
     * Create the actual DOM element for a vehicle
     */
    _createVehicleElement(vehicleId, vehicleType, direction) {
        const img = document.createElement('img');

        // Basic properties
        img.src = this._getSpriteUrl(vehicleType);
        img.alt = `${vehicleType} vehicle`;
        img.classList.add('vehicle-sprite');
        img.dataset.vehicleId = vehicleId;
        img.dataset.vehicleType = vehicleType;

        // Position the vehicle based on direction only
        this._positionVehicle(img, direction);

        // Setup animation cleanup
        this._setupAnimationCleanup(img, vehicleId);

        return img;
    }
    
    /**
     * Position vehicle based on direction only
     * Right: lower lane (60% of viewport), Left: upper lane (30% of viewport)
     */
    _positionVehicle(element, direction = 'right') {
        // Calculate Y position based on DIRECTION only
        const topPosition = this._calculateLanePosition(direction);
        element.style.top = `${topPosition}px`;

        // Add CSS classes for animation - let CSS handle left positioning
        switch (direction) {
            case 'left':
                element.classList.add('flipped', 'move-left');
                break;

            case 'right':
                element.classList.add('move-right');
                break;

            case 'stationary':
                element.classList.add('stationary');
                element.style.left = '50%';
                element.style.transform = 'translateX(-50%)';
                break;

            default:
                element.classList.add('move-right');
        }
    }
    
    /**
     * Calculate Y position based on direction only
     * Right direction: lower lane (60% of viewport)
     * Left direction: upper lane (30% of viewport)
     */
    _calculateLanePosition(direction = 'right') {
        let containerHeight = this.container ? this.container.clientHeight : 0;

        // Fallback to window height if container has no height
        if (containerHeight === 0) {
            containerHeight = window.innerHeight - 60; // Subtract header height
        }

        // Direction-based positioning for collision avoidance
        if (direction === 'left') {
            return containerHeight * 0.30; // Upper lane
        } else {
            return containerHeight * 0.60; // Lower lane (default)
        }
    }
    
    /**
     * Get sprite URL with fallback handling
     */
    _getSpriteUrl(vehicleType) {
        const spriteMap = {
            'car': 'car.svg',
            'truck': 'truck.svg',
            'bus': 'bus.svg',
            'motorcycle': 'motorcycle.svg',
            'bicycle': 'bicycle.svg',
            'other_vehicle': 'other_vehicle.svg'
        };
        
        const filename = spriteMap[vehicleType] || 'other_vehicle.svg';
        return `assets/sprites/${filename}`;
    }
    
    /**
     * Setup animation end cleanup
     */
    _setupAnimationCleanup(element, vehicleId) {
        const handleAnimationEnd = () => {
            this._removeVehicle(vehicleId);
        };
        
        // Listen for animation end
        element.addEventListener('animationend', handleAnimationEnd);
        
        // Fallback cleanup after animation duration + buffer
        const timeoutId = setTimeout(() => {
            this._removeVehicle(vehicleId);
        }, this.animationDuration + 1000);
        
        // Store cleanup data for proper memory management
        this.vehicleCleanupData.set(vehicleId, {
            listener: handleAnimationEnd,
            timeout: timeoutId
        });
    }
    
    /**
     * Remove vehicle from DOM and tracking
     */
    _removeVehicle(vehicleId) {
        const element = this.activeVehicles.get(vehicleId);
        const cleanupData = this.vehicleCleanupData.get(vehicleId);
        
        if (element) {
            // Clean up event listeners to prevent memory leaks
            if (cleanupData && cleanupData.listener) {
                element.removeEventListener('animationend', cleanupData.listener);
            }
            
            // Clear timeout to prevent orphaned timers
            if (cleanupData && cleanupData.timeout) {
                clearTimeout(cleanupData.timeout);
            }
            
            // Remove from DOM
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
            
            // Clean up tracking data
            this.activeVehicles.delete(vehicleId);
            this.vehicleCleanupData.delete(vehicleId);
            
            console.log('Vehicle removed:', vehicleId);
        }
    }
    
    /**
     * Get current active vehicles count
     */
    getActiveVehiclesCount() {
        return this.activeVehicles.size;
    }
    
    /**
     * Clear all active vehicles (for cleanup)
     */
    clearAll() {
        this.activeVehicles.forEach((element, vehicleId) => {
            const cleanupData = this.vehicleCleanupData.get(vehicleId);
            
            // Clean up event listeners
            if (cleanupData && cleanupData.listener) {
                element.removeEventListener('animationend', cleanupData.listener);
            }
            
            // Clear timeouts
            if (cleanupData && cleanupData.timeout) {
                clearTimeout(cleanupData.timeout);
            }
            
            // Remove from DOM
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
        });
        
        this.activeVehicles.clear();
        this.vehicleCleanupData.clear();
        console.log('All vehicles cleared');
    }
}