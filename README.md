# TrafficMetry

A real-time traffic monitoring system that uses AI to detect and classify vehicles from an IP camera stream, processes the data through an advanced tracking pipeline, and provides visualization through a web interface.Built with Python, YOLO, and ByteTrack.


## About

TrafficMetry is a complete application built from the ground up – from hardware setup and AI integration to a live web-based frontend. The system is designed to run 24/7 on a mini PC and provides comprehensive traffic analysis capabilities.
This project was born from a practical need: to objectively measure and understand the traffic flow on a busy local street.

**Key Features:**
- **Real-time Vehicle Detection**: Continuously analyzes live video feed from IP cameras using YOLO AI models
- **Multi-class Classification**: Detects and classifies vehicles (car, truck, bus, motorcycle, bicycle)
- **Dynamic Direction Detection**: Intelligent movement analysis without requiring manual lane calibration
- **Advanced Vehicle Tracking**: ByteTrack-based tracking system for consistent vehicle journeys
- **Event-driven Architecture**: Processes vehicle lifecycle events (entered, updated, exited)
- **Performance Monitoring**: Comprehensive diagnostics with multi-threaded viewer
- **Data Persistence**: SQLite database for long-term traffic analysis
- **Future-ready**: WebSocket infrastructure for real-time web visualization

## Tech Stack

**Current Implementation:**

* **Backend:** Python 3.11+, FastAPI, OpenCV, SQLite
* **AI/ML:** Ultralytics YOLO, ByteTrack vehicle tracking
* **Frontend:** HTML, CSS, JavaScript (with planned WebSocket integration)
* **Development:** UV package manager, Ruff linting, MyPy type checking

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).

[![CC BY-NC-SA 4.0](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

This means you can share and adapt this work for non-commercial purposes, provided you give appropriate credit and distribute any derivative works under the same license.