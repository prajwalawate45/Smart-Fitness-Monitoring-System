import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from utils import calculate_angle, draw_rep_count

# Configure logging
logging.basicConfig(
    filename='rep_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_FEEDBACK_DURATION = 60  # ~2 seconds at 30 fps
DEFAULT_VISIBILITY_THRESHOLD = 0.5
DEFAULT_LOW_VISIBILITY_THRESHOLD = 0.15
DEFAULT_MIN_FRAMES_BETWEEN_REPS = 10
DEFAULT_ANGLE_BUFFER_SIZE = 5

# Exercise-specific constants...
SQUAT_KNEE_ANGLE_DOWN_CORRECT = 90
SQUAT_KNEE_ANGLE_UP = 160
SQUAT_BACK_ANGLE_MAX = 20
SQUAT_KNEE_AMPLITUDE_MIN_CORRECT = 70
SQUAT_KNEE_AMPLITUDE_MIN_COUNT = 20

BICEP_ELBOW_ANGLE_DOWN_CORRECT = 50
BICEP_ELBOW_ANGLE_UP = 160
BICEP_SHOULDER_ANGLE_MAX = 20
BICEP_ELBOW_AMPLITUDE_MIN_CORRECT = 110
BICEP_ELBOW_AMPLITUDE_MIN_COUNT = 40

OVERHEAD_ELBOW_ANGLE_DOWN = 100
OVERHEAD_ELBOW_ANGLE_UP_CORRECT = 170
OVERHEAD_ARM_ANGLE_MAX = 45
OVERHEAD_TORSO_ANGLE_MAX = 35
OVERHEAD_ELBOW_AMPLITUDE_MIN_CORRECT = 70
OVERHEAD_ELBOW_AMPLITUDE_MIN_COUNT = 20
OVERHEAD_HIP_Y_MAX_CHANGE = 100

class BaseRepCounter:
    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.is_correct = True
        self.rep_correct = True
        self.errors: List[str] = []
        self.feedback = ""
        self.feedback_frames = 0
        self.feedback_duration = DEFAULT_FEEDBACK_DURATION
        self.last_rep_snapshot: Optional[Dict[str, Any]] = None

    def _update_feedback(self, new_feedback: str) -> None:
        if new_feedback:
            self.feedback = new_feedback
            self.feedback_frames = self.feedback_duration
            logging.debug(f"{self.exercise_name} Feedback: {self.feedback}")
        elif self.feedback_frames > 0:
            self.feedback_frames -= 1
        else:
            self.feedback = ""

    def _check_visibility(self, landmarks: List[float], threshold: float = DEFAULT_VISIBILITY_THRESHOLD) -> bool:
        return len(landmarks) > 3 and landmarks[3] > threshold

    def _smooth_angle(self, angle: float, buffer: List[float], buffer_size: int = DEFAULT_ANGLE_BUFFER_SIZE) -> float:
        buffer.append(angle)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        return sum(buffer) / len(buffer)

class SquatCounter(BaseRepCounter):
    def __init__(self):
        super().__init__("Squat")
        self.state = "up"
        self.min_knee = float('inf')
        self.knee_angles: List[float] = []
        self.back_angles: List[float] = []
        self.frames_since_last_rep = DEFAULT_MIN_FRAMES_BETWEEN_REPS
        self.angle_buffer: List[float] = []

    def update(self, landmarks, image) -> Tuple[np.ndarray, str]:
        new_feedback = ""
        self.frames_since_last_rep += 1

        if not landmarks:
            self.is_correct = False
            self.errors.append("Missing landmarks")
            logging.debug("Squat: Missing landmarks")
            new_feedback = "Ensure full body is visible"
        else:
            h, w, _ = image.shape
            def get_point(idx): 
                lm = landmarks.landmark[idx]
                return [lm.x*w, lm.y*h, lm.z*w, lm.visibility]

            left_hip = get_point(23)
            left_knee = get_point(25)
            left_ankle = get_point(27)
            right_hip = get_point(24)
            right_knee = get_point(26)
            right_ankle = get_point(28)
            left_shoulder = get_point(11)

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            knee_angle = (left_knee_angle + right_knee_angle) / 2
            smoothed_knee = self._smooth_angle(knee_angle, self.angle_buffer)
            
            back_vector = (left_shoulder[0] - left_hip[0], left_shoulder[1] - left_hip[1])
            vertical_vector = (0, -1)
            back_angle = calculate_angle(back_vector, (0, 0), vertical_vector)

            self.knee_angles.append(smoothed_knee)
            self.back_angles.append(back_angle)

            frame_correct = (
                back_angle < SQUAT_BACK_ANGLE_MAX and
                self._check_visibility(left_knee) and
                self._check_visibility(right_knee)
            )
            if not frame_correct:
                if not self._check_visibility(left_knee) or not self._check_visibility(right_knee):
                    self.errors.append("Low knee visibility")
                    new_feedback = "Ensure knees are visible"
                elif back_angle >= SQUAT_BACK_ANGLE_MAX:
                    self.errors.append(f"Back angle too large: {back_angle:.1f}°")
                    new_feedback = "Keep back straight"
                self.rep_correct = False
            self.is_correct = frame_correct

            # Robust state machine
            if self.state == "up" and smoothed_knee <= SQUAT_KNEE_ANGLE_DOWN_CORRECT:
                self.state = "down"
                self.min_knee = smoothed_knee
                self.rep_correct = frame_correct
            elif self.state == "down" and smoothed_knee >= SQUAT_KNEE_ANGLE_UP and self.frames_since_last_rep > DEFAULT_MIN_FRAMES_BETWEEN_REPS:
                amplitude = SQUAT_KNEE_ANGLE_UP - self.min_knee
                if amplitude >= SQUAT_KNEE_AMPLITUDE_MIN_COUNT:
                    if amplitude >= SQUAT_KNEE_AMPLITUDE_MIN_CORRECT and self.min_knee <= SQUAT_KNEE_ANGLE_DOWN_CORRECT and self.rep_correct:
                        self.correct_reps += 1
                        new_feedback = "Great job!"
                    else:
                        self.incorrect_reps += 1
                        if self.min_knee > SQUAT_KNEE_ANGLE_DOWN_CORRECT:
                            self.errors.append(f"Shallow squat: min knee {self.min_knee:.1f}° > {SQUAT_KNEE_ANGLE_DOWN_CORRECT}°")
                            new_feedback = "Bend knees more"
                        elif amplitude < SQUAT_KNEE_AMPLITUDE_MIN_CORRECT:
                            self.errors.append(f"Low knee amplitude: {amplitude:.1f}°")
                            new_feedback = "Squat deeper"
                    self.last_rep_snapshot = {
                        "exercise": "Squat",
                        "status": "correct" if self.rep_correct else "incorrect",
                        "knee_avg": sum(self.knee_angles)/max(len(self.knee_angles),1),
                        "back_avg": sum(self.back_angles)/max(len(self.back_angles),1),
                        "errors": list(self.errors)
                    }
                    self.rep_correct = True
                    self.errors = []
                    self.knee_angles = []
                    self.back_angles = []
                self.state = "up"
                self.min_knee = float('inf')
                self.frames_since_last_rep = 0

        self._update_feedback(new_feedback)
        return draw_rep_count(image, self.correct_reps, self.incorrect_reps), self.feedback

class BicepCurlCounter(BaseRepCounter):
    def __init__(self):
        super().__init__("Bicep Curl")
        self.state = "down"
        self.min_elbow = float('inf')
        self.elbow_angles: List[float] = []
        self.shoulder_angles: List[float] = []
        self.frames_since_last_rep = DEFAULT_MIN_FRAMES_BETWEEN_REPS
        self.angle_buffer: List[float] = []

    def update(self, landmarks, image) -> Tuple[np.ndarray, str]:
        new_feedback = ""
        self.frames_since_last_rep += 1
        if not landmarks:
            self.is_correct = False
            self.errors.append("Missing landmarks")
            logging.debug("Bicep Curl: Missing landmarks")
            new_feedback = "Ensure arms are visible"
        else:
            h, w, _ = image.shape
            def get_point(idx): 
                lm = landmarks.landmark[idx]
                return [lm.x*w, lm.y*h, lm.z*w, lm.visibility]

            left_shoulder = get_point(11)
            left_elbow = get_point(13)
            left_wrist = get_point(15)
            right_shoulder = get_point(12)
            right_elbow = get_point(14)
            right_wrist = get_point(16)

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            smoothed_elbow = self._smooth_angle(elbow_angle, self.angle_buffer)

            left_arm_vector = (left_shoulder[0] - left_elbow[0], left_shoulder[1] - left_elbow[1])
            vertical_vector = (0, -1)
            shoulder_angle = calculate_angle(left_arm_vector, (0, 0), vertical_vector)

            self.elbow_angles.append(smoothed_elbow)
            self.shoulder_angles.append(shoulder_angle)

            frame_correct = (
                shoulder_angle < BICEP_SHOULDER_ANGLE_MAX and
                self._check_visibility(left_elbow) and 
                self._check_visibility(right_elbow)
            )
            if not frame_correct:
                if not self._check_visibility(left_elbow) or not self._check_visibility(right_elbow):
                    self.errors.append("Low elbow visibility")
                    new_feedback = "Ensure elbows are visible"
                elif shoulder_angle >= BICEP_SHOULDER_ANGLE_MAX:
                    self.errors.append(f"Shoulder angle too large: {shoulder_angle:.1f}°")
                    new_feedback = "Lower your shoulders"
                self.rep_correct = False
            self.is_correct = frame_correct

            if self.state == "down" and smoothed_elbow >= BICEP_ELBOW_ANGLE_UP:
                self.state = "up"
                self.max_elbow = smoothed_elbow
                self.rep_correct = frame_correct
            elif self.state == "up" and smoothed_elbow <= BICEP_ELBOW_ANGLE_DOWN_CORRECT and self.frames_since_last_rep > DEFAULT_MIN_FRAMES_BETWEEN_REPS:
                amplitude = self.max_elbow - smoothed_elbow
                if amplitude >= BICEP_ELBOW_AMPLITUDE_MIN_COUNT:
                    if amplitude >= BICEP_ELBOW_AMPLITUDE_MIN_CORRECT and smoothed_elbow <= BICEP_ELBOW_ANGLE_DOWN_CORRECT and self.rep_correct:
                        self.correct_reps += 1
                        new_feedback = "Great job!"
                    else:
                        self.incorrect_reps += 1
                        if smoothed_elbow > BICEP_ELBOW_ANGLE_DOWN_CORRECT:
                            self.errors.append(f"Shallow curl: min elbow {smoothed_elbow:.1f}° > {BICEP_ELBOW_ANGLE_DOWN_CORRECT}°")
                            new_feedback = "Bend elbow more"
                        elif amplitude < BICEP_ELBOW_AMPLITUDE_MIN_CORRECT:
                            self.errors.append(f"Low elbow amplitude: {amplitude:.1f}°")
                            new_feedback = "Curl higher"
                    self.last_rep_snapshot = {
                        "exercise": "Bicep Curl",
                        "status": "correct" if self.rep_correct else "incorrect",
                        "elbow_avg": sum(self.elbow_angles)/max(len(self.elbow_angles),1),
                        "shoulder_avg": sum(self.shoulder_angles)/max(len(self.shoulder_angles),1),
                        "errors": list(self.errors)
                    }
                    self.rep_correct = True
                    self.errors = []
                    self.elbow_angles = []
                    self.shoulder_angles = []
                self.state = "down"
                self.min_elbow = float('inf')
                self.frames_since_last_rep = 0

        self._update_feedback(new_feedback)
        return draw_rep_count(image, self.correct_reps, self.incorrect_reps), self.feedback

class OverheadPressCounter(BaseRepCounter):
    def __init__(self):
        super().__init__("Overhead Press")
        self.state = "down"
        self.min_elbow = float('inf')
        self.max_elbow = float('-inf')
        self.elbow_angles: List[float] = []
        self.arm_angles: List[float] = []
        self.torso_angles: List[float] = []
        self.left_hip_ys: List[float] = []
        self.right_hip_ys: List[float] = []
        self.frames_since_last_rep = DEFAULT_MIN_FRAMES_BETWEEN_REPS
        self.angle_buffer: List[float] = []

    def update(self, landmarks, image) -> Tuple[np.ndarray, str]:
        new_feedback = ""
        self.frames_since_last_rep += 1
        if not landmarks:
            self.is_correct = False
            self.errors.append("Missing landmarks")
            logging.debug("Overhead Press: Missing landmarks")
            new_feedback = "Ensure full body is visible"
        else:
            h, w, _ = image.shape
            def get_point(idx): 
                lm = landmarks.landmark[idx]
                return [lm.x*w, lm.y*h, lm.z*w, lm.visibility]

            left_shoulder = get_point(11)
            left_elbow = get_point(13)
            left_wrist = get_point(15)
            right_shoulder = get_point(12)
            right_elbow = get_point(14)
            right_wrist = get_point(16)
            left_hip = get_point(23)
            right_hip = get_point(24)

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            smoothed_elbow = self._smooth_angle(elbow_angle, self.angle_buffer)

            left_arm_vector = (left_wrist[0] - left_shoulder[0], left_wrist[1] - left_shoulder[1])
            right_arm_vector = (right_wrist[0] - right_shoulder[0], right_wrist[1] - right_shoulder[1])
            vertical_vector = (0, -1)
            left_arm_angle = calculate_angle(left_arm_vector, (0, 0), vertical_vector)
            right_arm_angle = calculate_angle(right_arm_vector, (0, 0), vertical_vector)
            arm_angle = (left_arm_angle + right_arm_angle) / 2

            shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            torso_vector = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
            torso_angle = calculate_angle(torso_vector, (0, 0), vertical_vector)

            self.elbow_angles.append(smoothed_elbow)
            self.arm_angles.append(arm_angle)
            self.torso_angles.append(torso_angle)
            self.left_hip_ys.append(left_hip[1])
            self.right_hip_ys.append(right_hip[1])

            wrist_visibility = (
                self._check_visibility(left_wrist, DEFAULT_LOW_VISIBILITY_THRESHOLD) and
                self._check_visibility(right_wrist, DEFAULT_LOW_VISIBILITY_THRESHOLD)
            )
            frame_correct = (
                arm_angle < OVERHEAD_ARM_ANGLE_MAX and
                torso_angle < OVERHEAD_TORSO_ANGLE_MAX and
                wrist_visibility
            )
            if not frame_correct:
                if not wrist_visibility:
                    self.errors.append(f"Low wrist visibility: L={left_wrist[3]:.2f}, R={right_wrist[3]:.2f}")
                    new_feedback = "Ensure wrists are visible"
                elif torso_angle >= OVERHEAD_TORSO_ANGLE_MAX:
                    self.errors.append(f"Torso angle too large: {torso_angle:.1f}°")
                    new_feedback = "Keep torso upright"
                elif arm_angle >= OVERHEAD_ARM_ANGLE_MAX:
                    self.errors.append(f"Arm angle too large: {arm_angle:.1f}°")
                    new_feedback = "Align arms vertically"
                self.rep_correct = False
            self.is_correct = frame_correct

            # State machine for overhead press
            if self.state == "down" and smoothed_elbow >= OVERHEAD_ELBOW_ANGLE_UP_CORRECT:
                self.state = "up"
                self.max_elbow = smoothed_elbow
                self.rep_correct = frame_correct
            elif self.state == "up" and smoothed_elbow <= OVERHEAD_ELBOW_ANGLE_DOWN and self.frames_since_last_rep > DEFAULT_MIN_FRAMES_BETWEEN_REPS:
                amplitude = self.max_elbow - smoothed_elbow
                hips = self.left_hip_ys + self.right_hip_ys
                hip_y_change = max(hips or [0]) - min(hips or [0])
                if amplitude >= OVERHEAD_ELBOW_AMPLITUDE_MIN_COUNT:
                    if (amplitude >= OVERHEAD_ELBOW_AMPLITUDE_MIN_CORRECT and
                        self.max_elbow >= OVERHEAD_ELBOW_ANGLE_UP_CORRECT and
                        hip_y_change <= OVERHEAD_HIP_Y_MAX_CHANGE and
                        self.rep_correct):
                        self.correct_reps += 1
                        new_feedback = "Great job!"
                    else:
                        self.incorrect_reps += 1
                        if hip_y_change > OVERHEAD_HIP_Y_MAX_CHANGE:
                            self.errors.append(f"High hip movement: {hip_y_change:.1f}")
                            new_feedback = "Keep hips stable"
                        elif self.max_elbow < OVERHEAD_ELBOW_ANGLE_UP_CORRECT:
                            self.errors.append(f"Short press: max elbow {self.max_elbow:.1f}° < {OVERHEAD_ELBOW_ANGLE_UP_CORRECT}°")
                            new_feedback = "Raise arms higher"
                        elif amplitude < OVERHEAD_ELBOW_AMPLITUDE_MIN_CORRECT:
                            self.errors.append(f"Low elbow amplitude: {amplitude:.1f}°")
                            new_feedback = "Extend arms fully"
                    self.last_rep_snapshot = {
                        "exercise": "Overhead Press",
                        "status": "correct" if self.rep_correct else "incorrect",
                        "elbow_avg": sum(self.elbow_angles)/max(len(self.elbow_angles),1),
                        "arm_avg": sum(self.arm_angles)/max(len(self.arm_angles),1),
                        "torso_avg": sum(self.torso_angles)/max(len(self.torso_angles),1),
                        "hip_y_change": hip_y_change,
                        "errors": list(self.errors)
                    }
                    self.rep_correct = True
                    self.errors = []
                    self.elbow_angles = []
                    self.arm_angles = []
                    self.torso_angles = []
                    self.left_hip_ys = []
                    self.right_hip_ys = []
                self.state = "down"
                self.min_elbow = float('inf')
                self.max_elbow = float('-inf')
                self.frames_since_last_rep = 0

        self._update_feedback(new_feedback)
        return draw_rep_count(image, self.correct_reps, self.incorrect_reps), self.feedback
