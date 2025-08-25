# TrafficMetry

  A project to build a simple system for monitoring and analyzing local street traffic using an IP camera and AI 24/7

  ---

  ## About:

  This project is my approach to build a complete application – from real-time image processing, through backend API, to an interactive frontend.

  The goal is to create a system that can:
  - Continuously analyze a live video feed from an IP camera.
  - Detect vehicles in real-time using a YOLO AI model.
  - Classify vehicles into basic types (car, truck, bus, motorcycle, bicycle).
  - Determine each vehicle's lane and direction of travel.
  - Collect contextual data, such as local weather conditions, by connecting to external APIs.
  - Archive all this detailed data (timestamps, types, movement, weather) into a local SQLite database for long-term analysis.
  - Visualize the live flow of traffic on a simple, animated website using WebSockets.
  - Serve as a platform for future development, including training a custom AI model and building more advanced, procedural visualizations.

  The entire development process, from the first commits, is done publicly in this repository.

  ## Planned Tech Stack

  Initial plan uses these technologies:

  * **Backend:** Python, FastAPI, OpenCV, SQLite
  * **Frontend:** HTML, CSS, JavaScript
  * **Infrastructure:** System will run on a Mini PC with Linux (Ubuntu Server)

  *Note: Tech stack might evolve during project development.*

  ## Roadmap

  The project will be developed in several main phases. The plan below shows current development vision.

  * **Phase 1: MVP (Minimum Viable Product)**
      * **Goal:** Launch the first stable version that can detect vehicles and visualize them on a website
      * **Key tasks:**
          - Hardware and software setup
          - Basic vehicle detection and classification logic
          - Simple frontend with live visualization
          - Start collecting data

  * **Phase 2: Custom AI Model**
      * **Goal:** Make the system "smarter" by training a custom AI model that can recognize more detailed vehicle categories
      * **Key tasks:**
          - Build tools for labeling collected images
          - Train and deploy custom model

  * **Phase 3: Advanced Visualization**
      * **Goal:** Rebuild frontend into a procedural 2D engine that dynamically generates vehicle visualizations (e.g., taking their color into account)
      * **Key tasks:**
          - Add color detection in backend
          - Implement HTML Canvas rendering
