# 🏋️ Smart Fitness Monitoring System

## 📌 Introduction
The ** Smart Fitness Monitoring System** is a web-based fitness application that leverages **computer vision and artificial intelligence** to help users perform exercises correctly at home. The system provides **real-time posture analysis**, **automatic repetition counting**, and **performance feedback**, making workouts safer and more effective without the need for a personal trainer.

The application currently supports the following exercises:
- Squat
- Bicep Curl
- Overhead Press

---

## 🎯 Objectives
- To assist users in maintaining correct exercise posture
- To automatically count exercise repetitions
- To provide instant feedback for form correction
- To track workout progress over time

---

## ✨ Key Features

### 🔍 Real-Time Pose Detection
- Uses **MediaPipe** to detect body landmarks and joint positions.
- Tracks body movement accurately in real time.

### 🔢 Automatic Rep Counting
- Counts repetitions automatically for supported exercises.
- Differentiates between correct and incorrect repetitions based on joint angles and movement rules.

### 💬 Feedback System
- Provides rule-based feedback for posture correction.
- Optional AI-powered feedback using **Google Gemini API** (if API key is configured).

### 📊 Progress Tracking
- Stores workout data in a **SQLite database**.
- Displays progress using charts and performance metrics.

### 🖥️ Interactive User Interface
- Built using **Streamlit** for simplicity and responsiveness.
- Supports:
  - Live webcam input
  - Uploaded video files (MP4 format)

### 📘 Exercise Tutorials
- Provides basic tutorials and guidance for correct exercise techniques.

---

## 🧰 Technology Stack

- **Python 3.8+**
- **Streamlit** – Web application framework
- **MediaPipe** – Human pose estimation
- **OpenCV** – Video processing
- **SQLite** – Lightweight database
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Google Gemini API (Optional)** – AI-based feedback

~~~

## 📁 Project Structure

AI-Gym-Instructor/
│── main.py
│── pose_detector.py
│── rep_counter.py
│── utils.py
│── requirements.txt
│── workout_progress.db (generated automatically)
│── pages/
│ ├── progress.py
│ └── tutorials.py
~~~


## ▶️ Installation & Setup

### 1️⃣ Clone or Download the Project
Download the project files or clone the repository.

### 2️⃣ Create Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
Activate the environment:

Windows

.venv\Scripts\activate
Linux / macOS

source .venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
streamlit run main.py
The application will open automatically in your browser.

🧭 How to Use
Launch the application.

Select an exercise from the dropdown menu.

Choose input mode:

Webcam for live analysis

Upload Video for recorded workout analysis

(Optional) Enable Gemini AI Feedback if API key is configured.

Perform the exercise while monitoring:

Rep count

Form feedback

Progress data

Stop the session or navigate to progress and tutorial sections.

🧩 Module Description
main.py
Entry point of the application. Manages UI, exercise selection, and session control.

pose_detector.py
Handles pose detection and landmark extraction using MediaPipe.

rep_counter.py
Contains logic for repetition counting and form validation.

utils.py
Utility functions such as angle calculation and drawing helpers.

pages/progress.py
Displays workout statistics and visual progress charts.

pages/tutorials.py
Provides exercise tutorials and guidance.

requirements.txt
Lists all Python dependencies.

workout_progress.db
SQLite database used for storing workout history (created automatically).

🛠️ Development Phases
Pose detection integration using MediaPipe

Implementation of rep counting and form validation logic

Streamlit UI development

Optional AI feedback integration

Progress tracking and testing

🚀 Future Enhancements
Support for additional exercises (Push-ups, Deadlifts, Lunges)

Mobile application support

Cloud-based data storage and synchronization

Voice-based feedback system

Multi-language support

