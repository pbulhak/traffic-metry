/**
 * VehicleRenderer - SVG Sprite Visualization Engine
 * Manages dynamic creation, positioning and animation of vehicle sprites
 */

class VehicleRenderer {
    constructor(containerSelector = '.traffic-visualization') {
        this.container = document.querySelector(containerSelector);
        this.activeVehicles = new Map(); // vehicleId -> DOM element
        this.laneHeight = 60; // pixels per lane
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
            movement = { lane: 1, direction: 'right' }
        } = eventData;
        
        // Prevent duplicate vehicles
        if (this.activeVehicles.has(vehicleId)) {
            console.log('Vehicle already exists, skipping:', vehicleId);
            return;
        }
        
        // Performance limit check
        if (this.activeVehicles.size >= this.maxVehicles) {
            console.warn('Max vehicles limit reached, skipping new vehicle');
            return;
        }
        
        try {
            // Create vehicle element
            const vehicleElement = this._createVehicleElement(vehicleId, vehicleType, movement);
            
            // Add to container and tracking
            this.container.appendChild(vehicleElement);
            this.activeVehicles.set(vehicleId, vehicleElement);
            
            console.log(`Vehicle created: ${vehicleType} in lane ${movement.lane}, direction: ${movement.direction}`);
            
        } catch (error) {
            console.error('Error creating vehicle:', error, eventData);
        }
    }
    
    /**
     * Create the actual DOM element for a vehicle
     */
    _createVehicleElement(vehicleId, vehicleType, movement) {
        const img = document.createElement('img');
        
        // Basic properties
        img.src = this._getSpriteUrl(vehicleType);
        img.alt = `${vehicleType} vehicle`;
        img.classList.add('vehicle-sprite');
        img.dataset.vehicleId = vehicleId;
        img.dataset.vehicleType = vehicleType;
        
        // Position the vehicle
        this._positionVehicle(img, movement);
        
        // Setup animation cleanup
        this._setupAnimationCleanup(img, vehicleId);
        
        return img;
    }
    
    /**
     * Position vehicle based on lane and direction
     */
    _positionVehicle(element, movement) {
        const { lane = 1, direction = 'right' } = movement;
        
        // Calculate Y position based on lane
        const topPosition = this._calculateLanePosition(lane);
        element.style.top = `${topPosition}px`;
        
        // Set animation and initial position based on direction
        switch (direction) {
            case 'left':
                element.classList.add('flipped', 'move-left');
                element.style.left = 'calc(100% + 50px)'; // Start from right
                break;
                
            case 'right':
                element.classList.add('move-right');
                element.style.left = '-50px'; // Start from left
                break;
                
            case 'stationary':
                element.classList.add('stationary');
                element.style.left = '50%'; // Center
                element.style.transform = 'translateX(-50%)';
                break;
                
            default:
                // Default to right movement
                element.classList.add('move-right');
                element.style.left = '-50px';
        }
    }
    
    /**
     * Calculate Y position for a given lane number
     */
    _calculateLanePosition(lane) {
        // Ensure lane is a valid number
        const laneNumber = Math.max(1, parseInt(lane) || 1);
        
        // Base offset + (lane - 1) * lane height
        const baseOffset = 60; // Top margin
        return baseOffset + ((laneNumber - 1) * this.laneHeight);
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
        setTimeout(() => {
            this._removeVehicle(vehicleId);
        }, this.animationDuration + 1000);
    }
    
    /**
     * Remove vehicle from DOM and tracking
     */
    _removeVehicle(vehicleId) {
        const element = this.activeVehicles.get(vehicleId);
        
        if (element && element.parentNode) {
            element.parentNode.removeChild(element);
            this.activeVehicles.delete(vehicleId);
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
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
        });
        this.activeVehicles.clear();
        console.log('All vehicles cleared');
    }
}