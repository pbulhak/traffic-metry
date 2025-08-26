# Camera Calibration Guide for TrafficMetry

This guide explains how to calibrate your camera to define horizontal traffic lanes and assign movement directions for accurate vehicle detection and tracking.

## Overview

The TrafficMetry calibration tool allows you to:
- Define horizontal lane divider lines on your camera view
- Assign traffic directions (`left`, `right`, `stationary`) to each lane
- Save configuration for use by the main traffic monitoring system
- Test and validate your setup before production use

## Prerequisites

### Hardware Requirements
- IP camera with RTSP stream capability
- Network connectivity between camera and TrafficMetry system
- Camera positioned to capture side view of traffic (horizontal lanes)

### Software Requirements
- TrafficMetry system installed and configured
- Camera accessible via RTSP URL (configured in `.env`)
- Display server available for OpenCV GUI (if running locally)

### Camera Setup Recommendations
- **Viewing angle**: Side view capturing horizontal traffic lanes
- **Height**: Elevated position for clear lane separation
- **Stability**: Secure mounting to prevent camera movement
- **Lighting**: Adequate illumination for clear line visibility
- **Resolution**: 1920x1080 or higher for precise calibration

## Configuration

### Environment Setup
Ensure your `.env` file contains proper camera configuration:

```bash
# Camera Configuration
CAMERA__URL=rtsp://192.168.1.100:554/stream
CAMERA__WIDTH=1920
CAMERA__HEIGHT=1080
CAMERA__FPS=30
```

### Verify Camera Connection
Test camera connectivity before calibration:

```bash
python scripts/test_camera.py
```

If successful, you should see live video feed. Press 'q' to quit.

## Calibration Process

### Starting Calibration
Run the interactive calibration tool:

```bash
python scripts/calibrate.py
```

The tool will:
1. Connect to your camera
2. Capture a reference frame
3. Display the calibration interface

### Phase 1: Drawing Lane Dividers

#### Objective
Define horizontal lines that separate traffic lanes in your camera view.

#### Controls
- **Left Click**: Add line points (first click = start, second click = end)
- **Right Click**: Remove last line or cancel current line
- **Mouse Move**: Preview line while drawing
- **Enter**: Proceed to direction assignment (requires ≥2 lines)
- **Q**: Quit without saving

#### Instructions
1. **Identify Lane Boundaries**: Look for visible lane markers, road edges, or natural boundaries
2. **Click Start Point**: Left-click where you want the lane divider line to begin
3. **Click End Point**: Left-click where the line should end (completes the line)
4. **Repeat**: Add more lines to define all lane boundaries
5. **Review**: Check that lines properly separate your intended traffic lanes

#### Best Practices
- **Horizontal Lines**: Draw lines that follow the natural lane divisions
- **Full Width**: Lines should span the visible road area
- **Even Spacing**: Try to capture lanes of similar width
- **Minimum 2 Lines**: You need at least 2 lines to create 1 lane
- **Dashed Lines**: For broken lane markers, draw straight lines approximating the center

#### Troubleshooting Phase 1
| Problem | Solution |
|---------|----------|
| Can't see lane markings clearly | Adjust camera position or lighting |
| Lines appear crooked | Camera may be tilted - check mounting |
| Wrong line placement | Right-click to remove last line |
| Lines too close together | Space them further apart for better separation |

### Phase 2: Assigning Traffic Directions

#### Objective  
Assign movement directions to the horizontal lanes created between your divider lines.

#### Controls
- **Left Click in Lane**: Cycle direction (stationary → left → right → stationary)
- **S**: Save configuration and exit
- **R**: Reset entire calibration
- **Q**: Quit without saving

#### Direction Meanings
- **🟢 LEFT**: Traffic moves from right to left on screen
- **🔴 RIGHT**: Traffic moves from left to right on screen  
- **🟡 STATIONARY**: No traffic movement expected (parking, shoulder)

#### Visual Indicators
- **Green overlay + LEFT ←**: Lane configured for leftward traffic
- **Red overlay + RIGHT →**: Lane configured for rightward traffic
- **Yellow overlay + STATIONARY**: Lane configured as non-moving area

#### Instructions
1. **Review Lanes**: Observe the colored overlays between your divider lines
2. **Click Each Lane**: Click in the center of each lane area
3. **Cycle Directions**: Each click cycles through stationary → left → right → stationary
4. **Verify Colors**: Ensure colors match your intended traffic flow
5. **Save Configuration**: Press 's' to save and exit

#### Best Practices
- **Observe Traffic**: If possible, watch actual traffic to verify directions
- **Bidirectional Roads**: Assign appropriate directions to each lane
- **Edge Lanes**: Consider shoulders/parking as stationary
- **Double-Check**: Review all assignments before saving

#### Troubleshooting Phase 2
| Problem | Solution |
|---------|----------|
| Can't click in lane | Ensure click is within lane boundaries |
| Wrong direction assigned | Click lane again to cycle to correct direction |
| Lane too narrow | Recalibrate with better line spacing |
| Colors hard to see | Check display settings and lighting |

## Output and Integration

### Configuration File
Successful calibration creates `config.ini` with the following structure:

```ini
[lanes]
line_1 = x1,y1,x2,y2
line_2 = x1,y1,x2,y2
line_3 = x1,y1,x2,y2

[directions]
0 = left
1 = right
2 = stationary
```

### File Location
- **Development**: `config.ini` in project root
- **Production**: Specify path in `CONFIG_FILE` environment variable

### Validation
The system automatically validates your configuration:
- ✅ Minimum 2 divider lines required
- ✅ Proper coordinate format
- ✅ Valid direction values (`left`, `right`, `stationary`)
- ✅ Lane-to-direction mapping consistency

### Integration with Main System
The main TrafficMetry system automatically loads `config.ini` to:
- Define detection regions for each traffic lane
- Apply appropriate movement classifications
- Filter and validate vehicle detections
- Generate accurate traffic analytics

## Advanced Usage

### Recalibration
To update existing configuration:
1. Run calibration tool (overwrites existing `config.ini`)
2. Backup existing configuration if needed:
   ```bash
   cp config.ini config.ini.backup
   ```

### Multiple Camera Setup
For multiple cameras:
1. Calibrate each camera separately
2. Save configurations with unique names:
   ```bash
   mv config.ini config_camera1.ini
   ```
3. Update `CONFIG_FILE` environment variable per camera

### Testing Configuration
After calibration, test with main system:
```bash
# Run traffic detection with new calibration
python main.py
```

## Troubleshooting

### Camera Connection Issues
| Problem | Symptoms | Solution |
|---------|----------|----------|
| Camera not accessible | "Camera connection failed" error | Check network, RTSP URL, credentials |
| Poor image quality | Blurry or dark reference frame | Adjust camera focus, lighting, settings |
| Connection timeout | Tool hangs on connection | Verify camera IP, port, network path |

### Calibration Interface Issues  
| Problem | Symptoms | Solution |
|---------|----------|----------|
| No GUI window | Tool runs but no display | Check X11 forwarding, display server |
| Mouse clicks not working | Clicks ignored | Ensure window has focus, try different area |
| Lines appear wrong | Unexpected line placement | Check camera orientation, mounting |

### Configuration Issues
| Problem | Symptoms | Solution |
|---------|----------|----------|
| Can't save config | "Permission denied" error | Check write permissions, disk space |
| Invalid configuration | Validation errors | Review line placement, direction assignments |
| Config not loaded | Main system ignores calibration | Verify file path, format, syntax |

### Performance Issues
| Problem | Symptoms | Solution |
|---------|----------|----------|
| Slow response | Laggy interface | Check system resources, camera FPS |
| Memory errors | Tool crashes | Reduce camera resolution, restart system |
| Display problems | Corrupted graphics | Update graphics drivers, check OpenCV |

## Best Practices Summary

### Camera Setup
- ✅ Position for clear horizontal lane view
- ✅ Secure mounting prevents movement
- ✅ Adequate lighting for line visibility
- ✅ Test RTSP connection before calibration

### Line Drawing
- ✅ Follow natural lane boundaries
- ✅ Draw straight approximations for dashed lines  
- ✅ Space lines evenly for consistent lanes
- ✅ Use right-click to correct mistakes

### Direction Assignment
- ✅ Observe real traffic for verification
- ✅ Consider all lane types (traffic, parking, shoulder)
- ✅ Double-check assignments before saving
- ✅ Test configuration with main system

### Maintenance
- ✅ Recalibrate after camera adjustments
- ✅ Backup configurations before changes
- ✅ Document camera-specific settings
- ✅ Regular validation with traffic data

## Integration Examples

### Basic Integration
```python
from backend.config import get_config

# Load calibrated configuration
config = get_config()
lane_config = config.lanes

if lane_config:
    print(f"Loaded {len(lane_config.lines)} lane dividers")
    print(f"Direction assignments: {lane_config.directions}")
else:
    print("No calibration found - run calibration tool")
```

### Lane Detection Usage
```python
from backend.config import get_config

def classify_vehicle_lane(bbox, frame_shape):
    """Classify vehicle lane based on calibration."""
    config = get_config()
    
    if not config.lanes:
        return None, "uncalibrated"
    
    # Use calibrated lanes for classification
    vehicle_center_y = (bbox[1] + bbox[3]) // 2
    
    # Match against calibrated lane boundaries
    # Implementation would use config.lanes.lines and .directions
    
    return lane_id, direction
```

## Support

For additional help:
- Check the main [README.md](../README.md) for system requirements
- Review [MCP.md](../MCP.md) for technical specifications  
- Submit issues to the project repository
- Consult system logs for detailed error messages

---

*This documentation is part of the TrafficMetry traffic monitoring system. For technical details, see the project documentation in the `docs/` directory.*