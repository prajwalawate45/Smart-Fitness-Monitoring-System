🏋️ Smart Fitness Monitoring System
📌 Overview

The Smart Fitness Monitoring System is a web-based application designed to enhance home workouts using computer vision and artificial intelligence. It provides real-time exercise form correction, rep counting, and progress tracking to help users exercise safely and effectively.

The system currently supports three exercises:

Squat

Bicep Curl

Overhead Press

Users receive instant feedback, performance metrics, and visual progress reports through an interactive web interface.

✨ Features
🔍 Real-Time Form Analysis

Uses MediaPipe for human pose estimation.

Detects body landmarks and joint angles in real time.

🔢 Intelligent Rep Counting

Automatically counts repetitions.

Differentiates between correct and incorrect reps.

Applies exercise-specific rules (e.g., squat knee angle ≤ 90°).

💬 Feedback System

Rule-based feedback for posture correction.

Optional AI-enhanced feedback using the Google Gemini API (if enabled).

📊 Progress Tracking

Stores workout data in a SQLite database.

Displays progress using charts and performance metrics.

🖥️ User-Friendly Interface

Built with Streamlit.

Supports:

Live webcam input

Uploaded video files (MP4)

📘 Exercise Tutorials

Static guides explaining correct exercise techniques.

🧰 Tech Stack

Python 3.8+

Streamlit – Web UI framework

MediaPipe – Pose detection and landmarks

OpenCV – Video processing

SQLite – Local database

NumPy & Matplotlib – Data processing and visualization

Google Gemini API (Optional) – Advanced AI feedback

▶️ How to Run the Project
1️⃣ Project Setup

Create the following folder structure:

AI-Gym-Instructor/
│── main.py
│── pose_detector.py
│── rep_counter.py
│── utils.py
│── requirements.txt
│── workout_progress.db   (auto-generated)
│── pages/
│   ├── progress.py
│   └── tutorials.py


📌 The pages folder is required for Streamlit navigation.

2️⃣ Install Dependencies

Open your terminal in the project directory and run:

pip install -r requirements.txt

3️⃣ Run the Application

Start the Streamlit app using:

streamlit run main.py

🧭 Usage Guide

Select an exercise from the dropdown menu.

Choose one of the input methods:

Webcam (real-time analysis)

Upload Video (MP4 file)

(Optional) Enable Gemini AI Feedback if the API key is configured.

View:

Live feedback and rep count

Progress charts

Exercise tutorials

Stop the session or review annotated video output.

🧩 Project Modules

main.py – Application entry point and UI controller

pose_detector.py – MediaPipe pose detection logic

rep_counter.py – Rep counting and form validation

utils.py – Utility functions (angle calculation, drawing helpers)

progress.py – Displays workout statistics and charts

tutorials.py – Exercise guidance and instructions

requirements.txt – Project dependencies

workout_progress.db – SQLite database (generated on first run)

.env (optional) – Stores Gemini API key

🛠️ Development Process

Phase 1: Integrated MediaPipe pose detection

Phase 2: Implemented rep counting and form logic

Phase 3: Built Streamlit UI with webcam and video support

Phase 4: Added optional Gemini AI feedback with fallbacks

Phase 5: Implemented progress tracking and final testing

🚀 Future Enhancements

Add more exercises (Push-Up, Deadlift, Lunges)

Mobile application support (Kivy / Flutter)

Cloud-based data sync (Firebase)

Voice feedback system

Multi-language support
