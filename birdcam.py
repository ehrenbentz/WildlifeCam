#!/usr/bin/python3

from picamera2 import Picamera2
import numpy as np
from flask import Flask, render_template, Response
import cv2
from adafruit_servokit import ServoKit
import time
import threading
from queue import Queue
import datetime
import logging
import sys
import os
import shutil
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Record the script start time
script_start_time = datetime.datetime.now()

# Define servo limits
V_MIN = 20      # For Vertical servo (Greater values are DOWN)
V_MAX = 166     # For Vertical servo (Greater values are DOWN)
H_MIN = 1       # For Horizontal servo
H_MAX = 269     # For Horizontal servo

# Initialize global angle variables
global V_angle
global H_angle
V_angle = None
H_angle = None

# Threshold for angle comparisons
epsilon = 0.1

## Static servo settings
kit = ServoKit(channels=16)
Horizontal = 0
Vertical = 1
kit.servo[Vertical].actuation_range = 270
kit.servo[Horizontal].actuation_range = 270
kit.servo[Vertical].set_pulse_width_range(500, 2500)
kit.servo[Horizontal].set_pulse_width_range(500, 2500)

# Set maximum servo speeds (degrees per second)
V_max_speed = 1.0  # Vertical servo speed
H_max_speed = 1.0  # Horizontal servo speed

# Adjusted sleep duration between steps (seconds)
step_sleep = 0.01  # 10 milliseconds for smoother movement

# Calculated step sizes based on max speeds and sleep duration
V_step_size = 0.2  # Small step size for smoothness
H_step_size = 0.2  # Small step size for smoothness

# Set distance that the camera will move for each time the button is pressed
V_dist = 5
H_dist = 10

# Panning state variables
is_panning = False
last_panning_time = None
panning_lock = threading.Lock()

# Command queue for servo movements
servo_command_queue = Queue()

# Lock for resetting first_frame
frame_lock = threading.Lock()
reset_first_frame = False  # Flag to indicate first_frame needs to be reset
reset_first_frame_event = threading.Event()

def clamp(value, min_value, max_value):
    """Clamp the value between min_value and max_value."""
    return max(min_value, min(value, max_value))

def set_vertical_angle(angle):
    """Set the vertical servo angle within defined limits."""
    global V_angle
    angle = clamp(angle, V_MIN, V_MAX)
    kit.servo[Vertical].angle = angle
    V_angle = angle
    logging.debug(f"Vertical servo set to {V_angle} degrees.")

def set_horizontal_angle(angle):
    """Set the horizontal servo angle within defined limits."""
    global H_angle
    angle = clamp(angle, H_MIN, H_MAX)
    kit.servo[Horizontal].angle = angle
    H_angle = angle
    logging.debug(f"Horizontal servo set to {H_angle} degrees.")

def detach_servos():
    """Detach both vertical and horizontal servos to stop sending PWM signals."""
    kit.servo[Vertical].angle = None
    kit.servo[Horizontal].angle = None
    logging.debug("Servos detached to prevent jitter.")

def move_servo(current_angle, target_angle, set_angle_func, step_size, sleep_time):
    """
    Move a servo from current_angle to target_angle in controlled steps.

    Args:
        current_angle (float): The current angle of the servo.
        target_angle (float): The desired target angle.
        set_angle_func (function): Function to set the servo angle.
        step_size (float): Maximum degrees to move per step.
        sleep_time (float): Time to sleep between steps in seconds.
    """
    delta = target_angle - current_angle
    direction = 1 if delta > 0 else -1
    delta = abs(delta)

    while delta > epsilon:
        move = min(step_size, delta) * direction
        current_angle += move
        set_angle_func(current_angle)
        time.sleep(sleep_time)
        delta -= step_size

    # Ensure the final angle is set accurately
    set_angle_func(target_angle)
    detach_servos()

def check_and_initialize_servo_positions():
    """
    Directly move the servos to the center position (90 for vertical, 135 for horizontal).
    """
    global V_angle, H_angle

    # Set initial target positions (centered)
    initial_V_angle = 163
    initial_H_angle = 135

    # Move servos directly to the center position
    kit.servo[Vertical].angle = initial_V_angle
    V_angle = initial_V_angle
    logging.info(f"Vertical servo moved to {initial_V_angle} degrees.")

    kit.servo[Horizontal].angle = initial_H_angle
    H_angle = initial_H_angle
    logging.info(f"Horizontal servo moved to {initial_H_angle} degrees.")

def shutdown():
    """
    Move servos to the starting position (H = 135, V = 163) and shut down the script.
    """
    logging.info("Shutting down: Moving servos to start position...")

    # Move servos to the starting position
    move_servo(H_angle, 135, set_horizontal_angle, H_step_size, step_sleep)
    move_servo(V_angle, 163, set_vertical_angle, V_step_size, step_sleep)

    logging.info("Servos moved to start position. Shutting down...")

    # Properly stop the camera
    if picam2:
        picam2.stop()
        picam2.close()
        # Exit the script
        sys.exit()

@app.route('/shutdown')
def shutdown_route():
    """
    Route to shut down the server and move servos to the starting position.
    """
    servo_command_queue.put('shutdown')
    return ('', 204)

def servo_worker():
    while True:
        command = servo_command_queue.get()
        # Execute panning and tilting commands
        if command == 'up':
            up()
        elif command == 'down':
            down()
        elif command == 'left':
            left()
        elif command == 'right':
            right()
        elif command == 'center':
            re_center()
        elif command == 'shutdown':
            shutdown()
        servo_command_queue.task_done()

# Route to handle commands going back to the Pi
@app.route('/control/<command>')
def control(command):
    if command in ['up', 'down', 'left', 'right', 'center']:
        servo_command_queue.put(command)
    return ('', 204)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

# Global variables for camera streaming
picam2 = None
output_frame = None

def initialize_video_writer(frame, fps, video_filename):
    """
    Initialize the VideoWriter object with the given frame size and frame rate.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        video_filename,
        fourcc,
        fps,
        (frame.shape[1], frame.shape[0])
    )
    return video_writer

def capture_frames():
    global picam2, output_frame, frame_lock, reset_first_frame_event
    try:
        # Set a fixed frame ratefor use throughout capture_frames()
        fixed_frame_rate = 15
        # Configure picamera2
        config = picam2.create_video_configuration(main={"size": (1600, 900)})
        config["controls"] = {
            "Contrast": 1.0,
            "Saturation": 0.0,
            "Sharpness": 1.0,
            "AwbEnable": False,
            "FrameRate": fixed_frame_rate,
        }
        picam2.configure(config)
        picam2.start()

        # Initialize variables for motion detection and recording
        first_frame = None
        first_frame_time = None
        motion_detected = False
        recording = False
        recording_start_time = None
        recording_duration_seconds = 60
        frames_to_record = int(recording_duration_seconds * fixed_frame_rate)
        frames_recorded = 0
        video_writer = None

        # Initialize cooldown variables
        cooldown_duration = 10
        last_recording_end_time = None
        panning_cooldown_duration = 5

        # Initialize motion counter
        motion_counter = 0
        motion_threshold = 5

        # Frame rate calculation variables
        frame_times = []

        # Cooldown state variables
        panning_cooldown_passed = True
        recording_cooldown_passed = True

        while True:
            start_time = time.time()

            # Capture frame from camera
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Convert frame to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            current_time = datetime.datetime.now()

            # Check panning state
            with panning_lock:
                panning = is_panning
                last_panning = last_panning_time

            # Handle panning cooldown
            if last_panning is not None and not panning:
                time_since_panning = (current_time - last_panning).total_seconds()
                if time_since_panning >= panning_cooldown_duration:
                    if not panning_cooldown_passed:
                        print("Panning cooldown has passed. Ready to record movement...")
                        panning_cooldown_passed = True
                else:
                    panning_cooldown_passed = False

            # Reset first_frame after panning cooldown
            if reset_first_frame_event.is_set() and last_panning is not None:
                time_since_panning = (current_time - last_panning).total_seconds()
                if time_since_panning >= panning_cooldown_duration:
                    with frame_lock:
                        first_frame = gray
                        first_frame_time = current_time
                    reset_first_frame_event.clear()
                    continue

            # Initialize first_frame if None
            if first_frame is None:
                first_frame = gray
                first_frame_time = current_time
                continue

            # Compute the absolute difference between the current frame and first frame
            frame_delta = cv2.absdiff(first_frame, gray)
            thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)

            # Find contours on thresholded image
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                motion_detected = True
                break

            # Update first_frame periodically
            time_since_first_frame = (current_time - first_frame_time).total_seconds()
            if time_since_first_frame >= 30:
                first_frame = gray
                first_frame_time = current_time

            # Prevent motion counting during panning and cooldown
            if not panning and (last_panning is None or (current_time - last_panning).total_seconds() >= panning_cooldown_duration):
                if motion_detected:
                    motion_counter += 1
                else:
                    motion_counter = 0

            # Check if recording cooldown has passed
            if last_recording_end_time is not None:
                time_since_last_recording = (current_time - last_recording_end_time).total_seconds()
                if time_since_last_recording >= cooldown_duration:
                    if not recording_cooldown_passed:
                        print("Recording cooldown has passed. Ready to record movement...")
                        recording_cooldown_passed = True
                else:
                    recording_cooldown_passed = False

            # Start recording if motion is detected
            if (motion_counter >= motion_threshold and not recording and not panning and
                    recording_cooldown_passed and panning_cooldown_passed):
                # Start recording
                recording = True
                recording_start_time = current_time
                timestamp = recording_start_time.strftime("%H-%M-%S_%b-%d-%Y")
                date_today = recording_start_time.strftime("%Y-%m-%d")
                folder_path = f"/media/pi/Opossum/{date_today}"

                # Ensure folder exists
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    logging.info(f"Created folder: {folder_path}")

                video_filename = f"{folder_path}/{timestamp}.mp4"

                # Initialize video writer with fixed frame rate
                video_writer = initialize_video_writer(frame, fixed_frame_rate, video_filename)
                print(f"Motion detected! Started recording: {video_filename}")
                frames_recorded = 0
                motion_counter = 0
                recording_cooldown_passed = False

            # Write frames to video file if recording
            if recording:
                video_writer.write(frame)
                frames_recorded += 1
                # Stop recording after specified number of frames
                if frames_recorded >= frames_to_record:
                    recording = False
                    video_writer.release()
                    video_writer = None
                    last_recording_end_time = current_time
                    frames_recorded = 0  # Reset the frame counter
                    print(f"Recording stopped: {video_filename}")

            # Calculate frame processing time
            frame_duration = time.time() - start_time
            frame_times.append(frame_duration)

            with frame_lock:
                # Use the original frame for the live stream
                output_frame = frame.copy()

    except Exception as e:
        logging.error(f"Error in capture_frames: {e}")
        # Attempt to recover from the error
        try:
            picam2.stop()
            picam2.close()
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {cleanup_error}")
        finally:
            time.sleep(2)
            picam2 = Picamera2()
            capture_frames()

def gen_picam2():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_picam2(), mimetype='multipart/x-mixed-replace; boundary=frame')

def up():
    global V_angle, is_panning, last_panning_time, reset_first_frame_event

    with panning_lock:
        is_panning = True  # Indicate panning is in progress

    logging.info("Tilting up...")

    if V_angle is None:  # Handle initial state where angle might be undefined
        V_angle = 90.0  # Set to a reasonable default if None

    target_angle = clamp(V_angle - V_dist, V_MIN, V_MAX)  # Calculate the new target angle, ensuring it's within limits

    move_servo(  # Move the servo smoothly to the target angle
        current_angle=V_angle,
        target_angle=target_angle,
        set_angle_func=set_vertical_angle,
        step_size=V_step_size,
        sleep_time=step_sleep
    )
    V_angle = target_angle  # Update the global angle variable

    with panning_lock:
        is_panning = False  # Indicate panning is complete
        last_panning_time = datetime.datetime.now()  # Record the time panning stopped

    reset_first_frame_event.set()  # Signal to reset the first frame for motion detection

    logging.info("Tilting up completed.")
    detach_servos()  # Detach servos to prevent jitter

def down():
    global V_angle, is_panning, last_panning_time, reset_first_frame_event
    with panning_lock:
        is_panning = True
    logging.info("Tilting down...")
    if V_angle is None:
        V_angle = 90.0
    target_angle = clamp(V_angle + V_dist, V_MIN, V_MAX)

    move_servo(
        current_angle=V_angle,
        target_angle=target_angle,
        set_angle_func=set_vertical_angle,
        step_size=V_step_size,
        sleep_time=step_sleep
    )
    V_angle = target_angle
    print(V_angle)
    print(H_angle)

    with panning_lock:
        is_panning = False
        last_panning_time = datetime.datetime.now()
    reset_first_frame_event.set()
    logging.info("Tilting down completed.")
    detach_servos()

def left():
    global H_angle, is_panning, last_panning_time, reset_first_frame_event
    with panning_lock:
        is_panning = True
    logging.info("Panning left...")
    if H_angle is None:
        H_angle = 135.0
    target_angle = clamp(H_angle + H_dist, H_MIN, H_MAX)

    move_servo(
        current_angle=H_angle,
        target_angle=target_angle,
        set_angle_func=set_horizontal_angle,
        step_size=H_step_size,
        sleep_time=step_sleep
    )
    H_angle = target_angle

    with panning_lock:
        is_panning = False
        last_panning_time = datetime.datetime.now()
    reset_first_frame_event.set()
    logging.info("Panning left completed.")
    detach_servos()

def right():
    global H_angle, is_panning, last_panning_time, reset_first_frame_event
    with panning_lock:
        is_panning = True
    logging.info("Panning right...")
    if H_angle is None:
        H_angle = 135.0
    target_angle = clamp(H_angle - H_dist, H_MIN, H_MAX)

    move_servo(
        current_angle=H_angle,
        target_angle=target_angle,
        set_angle_func=set_horizontal_angle,
        step_size=H_step_size,
        sleep_time=step_sleep
    )
    H_angle = target_angle

    with panning_lock:
        is_panning = False
        last_panning_time = datetime.datetime.now()
    reset_first_frame_event.set()
    logging.info("Panning right completed.")
    detach_servos()

def re_center():
    global V_angle, H_angle, is_panning, last_panning_time, reset_first_frame_event
    with panning_lock:
        is_panning = True
    logging.info("Re-centering...")

    # Check if V_angle is None and set it to a safe midpoint if so
    if V_angle is None:
        V_angle = 95
        logging.info(f"Vertical angle was None. Setting to midpoint: {V_angle}.")

    # Center Vertical Servo
    target_V_angle = 95
    move_servo(
        current_angle=V_angle,
        target_angle=target_V_angle,
        set_angle_func=set_vertical_angle,
        step_size=V_step_size,
        sleep_time=step_sleep
    )
    V_angle = target_V_angle
    logging.info("Vertical servo reached center.")

    # Check if H_angle is None and set it to a safe midpoint if so
    if H_angle is None:
        H_angle = 135
        logging.info(f"Horizontal angle was None. Setting to midpoint: {H_angle}.")

    # Center Horizontal Servo
    target_H_angle = 135.0
    move_servo(
        current_angle=H_angle,
        target_angle=target_H_angle,
        set_angle_func=set_horizontal_angle,
        step_size=H_step_size,
        sleep_time=step_sleep
    )
    H_angle = target_H_angle
    logging.info("Horizontal servo reached center.")

    with panning_lock:
        is_panning = False
        last_panning_time = datetime.datetime.now()
    reset_first_frame_event.set()
    logging.info("Re-centering completed.")

    detach_servos()

if __name__ == '__main__':
    # Initialize Picamera2
    picam2 = Picamera2()

    # Check servo positions initially (ensure this happens before other operations)
    check_and_initialize_servo_positions()

    # Start the frame capture thread
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()

    # Start the servo worker thread
    servo_thread = threading.Thread(target=servo_worker)
    servo_thread.daemon = True
    servo_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
