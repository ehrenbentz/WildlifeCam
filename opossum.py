#!/usr/bin/python3

from picamera2 import Picamera2
import numpy as np
from flask import Flask, render_template, Response, request, send_from_directory
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


###################################################################
################# Camera / Streaming code #########################
###################################################################

# Global variables for camera streaming
picam2 = None
output_frame = None
recording = False
recording_lock = threading.Lock()

# Set camera resolution and FPS
resolution = (1920, 1080)
fps = 10
saturation = 0.0
brightness = 0.02

# Set motion detection sensitivity
abs_thresh = 30
contour_thresh = 600
reset_first_frame_seconds = 5

#  Recording settings
duration = 60

def record_video_with_hardware_acceleration(output_file, bitrate="5M"):
    """
    Record video using FFmpeg with hardware acceleration.
    """
    global resolution, fps, duration

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file

        # Input options
        "-f", "rawvideo",  # Input format is raw video
        "-pix_fmt", "yuv420p",  # Pixel format from OpenCV
        "-s", f"{resolution[0]}x{resolution[1]}",  # Frame size
        "-framerate", str(fps),  # Input frame rate
        "-i", "-",  # Read video from stdin

        # Output options (placed after inputs, before output file)
        "-c:v", "h264_v4l2m2m",  # Use hardware-accelerated H.264 encoder
        "-b:v", bitrate,  # Bitrate
        "-t", str(duration),  # Duration in seconds
        "-fps_mode", "passthrough",  # Output option for frame rate mode

        # Output file
        output_file
    ]

    # Start FFmpeg process
#    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
#    return ffmpeg_proc

    # Start FFmpeg process with suppressed output
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return ffmpeg_proc

def capture_frames():
    global picam2, output_frame, frame_lock, reset_first_frame_event, recording, recording_lock
    global resolution, fps, saturation, brightness, abs_thresh, contour_thresh

    ffmpeg_proc = None  # Store FFmpeg process during recording
    frame_count = 0  # Frame counter to track frames

    try:
        # Configure picamera2
        config = picam2.create_video_configuration(main={"size": resolution})
        config["controls"] = {
            "Contrast": 1.0,
            "Saturation": saturation,
            "Sharpness": 1.0,
            "AwbEnable": False,
            "FrameRate": fps,
            "Brightness": brightness,
        }
        picam2.configure(config)
        picam2.start()

        # Initialize variables for motion detection and recording
        first_frame = None
        first_frame_time = None
        motion_detected = False
        recording_start_time = None

        # Cooldown variables
        cooldown_duration = 1
        last_recording_end_time = None
        panning_cooldown_duration = 2
        motion_counter = 0
        motion_threshold = 5

        # Cooldown state variables
        panning_cooldown_passed = True
        recording_cooldown_passed = True

        while True:
            frame_count += 1  # Increment frame counter
            motion_detected = False

            # Capture frame from camera
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Get the current timestamp
            current_time = datetime.datetime.now()
            timestamp_text = current_time.strftime("%m-%d-%Y %H:%M:%S")  # MM-DD-YYYY HH:MM:SS

            # Convert frame to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # Check panning state (assuming these variables and locks are defined elsewhere)
            with panning_lock:
                panning = is_panning
                last_panning = last_panning_time

            # Handle panning cooldown
            if last_panning is not None and not panning:
                time_since_panning = (current_time - last_panning).total_seconds()
                panning_cooldown_passed = time_since_panning >= panning_cooldown_duration

            # Reset first_frame dynamically based on FPS
            if frame_count % int(reset_first_frame_seconds * fps) == 0:
                first_frame = gray
                first_frame_time = current_time

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
            thresh = cv2.threshold(frame_delta, abs_thresh, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=1)
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours on thresholded image
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Detect motion based on contours
            for c in contours:
                if cv2.contourArea(c) >= contour_thresh:
                    # Optionally draw bounding boxes
                    # x, y, w, h = cv2.boundingRect(c)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    motion_detected = True

            # Draw a gray rectangle behind timestamp
            x1, y1, x2, y2 = 20, 5, 405, 45
            roi = frame[y1:y2, x1:x2]
            gray_rect = np.full((y2 - y1, x2 - x1, 3), (90, 90, 90), dtype=np.uint8)
            alpha = 0.6  # Transparency factor
            blended_roi = cv2.addWeighted(gray_rect, alpha, roi, 1 - alpha, 0)
            frame[y1:y2, x1:x2] = blended_roi

            # Overlay the timestamp onto the frame
            cv2.putText(
                frame,
                timestamp_text,
                (25, 35),  # Position for text within the frame
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),  # Green color in BGR
                2
            )

            # Motion counting logic
            if motion_detected and not panning and panning_cooldown_passed and recording_cooldown_passed:
                motion_counter += 1
            else:
                motion_counter = 0

            # Check if recording cooldown has passed
            if last_recording_end_time is not None:
                time_since_last_recording = (current_time - last_recording_end_time).total_seconds()
                recording_cooldown_passed = time_since_last_recording >= cooldown_duration

            # Access the recording status safely
            with recording_lock:
                currently_recording = recording

            # Start recording if motion is detected
            if (motion_counter >= motion_threshold and not currently_recording and not panning and
                    recording_cooldown_passed and panning_cooldown_passed):
                with recording_lock:
                    recording = True
                recording_start_time = current_time
                timestamp = recording_start_time.strftime("%H-%M-%S_%m-%d-%Y")
                date_today = recording_start_time.strftime("%m-%d-%Y")
                folder_path = f"/home/WLCam/TMP_Videos/{date_today}"

                # Ensure folder exists
                os.makedirs(folder_path, exist_ok=True)

                video_filename = f"{folder_path}/{timestamp}.mp4"
                ffmpeg_proc = record_video_with_hardware_acceleration(video_filename)

                print(f"Motion detected! Started recording: {video_filename}")
                motion_counter = 0
                recording_cooldown_passed = False

            # Check if FFmpeg process has exited before writing frames
            if currently_recording and ffmpeg_proc:
                if ffmpeg_proc.poll() is not None:  # FFmpeg process has finished
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait()
                    ffmpeg_proc = None
                    last_recording_end_time = current_time
                    print(f"Recording stopped: {video_filename}")
                    with recording_lock:
                        recording = False
                else:
                    # FFmpeg is still running, write frames
                    # Convert frame to YUV420p before writing to FFmpeg
                    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                    try:
                        ffmpeg_proc.stdin.write(frame_yuv.tobytes())
                    except BrokenPipeError:
                        # FFmpeg has terminated unexpectedly; handle cleanup
                        ffmpeg_proc.stdin.close()
                        ffmpeg_proc.wait()
                        ffmpeg_proc = None
                        last_recording_end_time = current_time
                        with recording_lock:
                            recording = False
                        print(f"FFmpeg process terminated unexpectedly while recording {video_filename}")
                        continue

            # Use the original frame (with timestamp) for the live stream
            with frame_lock:
                output_frame = frame.copy()

    except Exception as e:
        logging.error(f"Error in capture_frames: {e}")
        # Attempt to recover from the error
        if ffmpeg_proc:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
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
            try:
                ret, jpeg = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ret:
                    continue
                frame = jpeg.tobytes()
            except Exception:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


### Serve webpage elements

# Initialize flask app
app = Flask(__name__)

# Initialize directories
VIDEO_DIR = "/home/WLCam/Datacube/Opossum/Videos"
THUMBNAIL_DIR = "/home/WLCam/Datacube/Opossum/Thumbnails"

@app.route('/')
def index():
    """Serve web page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Serve video stream"""
    return Response(gen_picam2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordings/')
def list_folders():
    try:
        folders = sorted([d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, d))], reverse=True)
        return render_template('folders.html', folders=folders)
    except Exception as e:
        return f"Error listing folders: {e}", 500

@app.route('/recordings/<folder>/')
def list_videos(folder):
    folder_path = os.path.join(VIDEO_DIR, folder)
    if not os.path.exists(folder_path):
        return "Folder not found", 404

    all_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.mp4')],
        reverse=request.args.get('order', 'desc') == 'desc'
    )

    return render_template(
        'videos.html',
        folder=folder,
        videos=all_files
    )

@app.route('/recordings/<folder>/<filename>')
def serve_video(folder, filename):
    folder_path = os.path.join(VIDEO_DIR, folder)
    return send_from_directory(folder_path, filename)

@app.route('/thumbnails/<folder>/<filename>')
def serve_thumbnail(folder, filename):
    thumbnail_dir = os.path.join(THUMBNAIL_DIR, folder)
    return send_from_directory(thumbnail_dir, filename)


###################################################################
################### Pan/Tilt Servo code ###########################
###################################################################

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
H_angle_center = 135
V_angle_center = 90

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
V_max_speed = 1.0
H_max_speed = 1.0

# Adjusted sleep duration between steps (seconds)
step_sleep = 0.01  # 10 milliseconds for smoother movement

# Calculated step sizes based on max speeds and sleep duration
V_step_size = 0.2  # Small step size for smoothness
H_step_size = 0.2  # Small step size for smoothness

# Set distance that the camera will move for each time the button is pressed
V_dist = 5
H_dist = 10

# Panning state variables for motion detection
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

def initialize_servo_positions():
    """
    Move the servos to the center position
    """
    global V_angle, H_angle, H_angle_center, V_angle_center

    # Set initial target positions (centered)
    initial_V_angle = V_angle_center
    initial_H_angle = H_angle_center

    # Move servos directly to the center position
    kit.servo[Vertical].angle = initial_V_angle
    V_angle = initial_V_angle
    logging.info(f"Vertical servo moved to {initial_V_angle} degrees.")

    kit.servo[Horizontal].angle = initial_H_angle
    H_angle = initial_H_angle
    logging.info(f"Horizontal servo moved to {initial_H_angle} degrees.")

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

def up():
    global V_angle, is_panning, last_panning_time, reset_first_frame_event
    with panning_lock:
        is_panning = True
    logging.info("Tilting up...")
    target_angle = clamp(V_angle - V_dist, V_MIN, V_MAX)
    move_servo(
        current_angle=V_angle,
        target_angle=target_angle,
        set_angle_func=set_vertical_angle,
        step_size=V_step_size,
        sleep_time=step_sleep
    )
    V_angle = target_angle
    with panning_lock:
        is_panning = False
        last_panning_time = datetime.datetime.now()
    reset_first_frame_event.set()
    logging.info("Tilting up completed.")
    detach_servos()

def down():
    global V_angle, is_panning, last_panning_time, reset_first_frame_event
    with panning_lock:
        is_panning = True
    logging.info("Tilting down...")
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
    global V_angle, H_angle, H_angle_center, V_angle_center, is_panning, last_panning_time, reset_first_frame_event
    with panning_lock:
        is_panning = True
    logging.info("Re-centering...")
    # Center Vertical Servo
    target_V_angle = V_angle_center
    move_servo(
        current_angle=V_angle,
        target_angle=target_V_angle,
        set_angle_func=set_vertical_angle,
        step_size=V_step_size,
        sleep_time=step_sleep
    )
    V_angle = target_V_angle
    logging.info("Vertical servo reached center.")
    # Center Horizontal Servo
    target_H_angle = H_angle_center
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

# Route to handle commands going back to the Pi
@app.route('/control/<command>')
def control(command):
    valid_commands = {'up', 'down', 'left', 'right', 'center'}
    if command not in valid_commands:
        return 'Invalid command', 400
    servo_command_queue.put(command)
    return '', 204

# Initialize Picamera2
picam2 = Picamera2()

# Move servos to center
initialize_servo_positions()

# Start the frame capture thread
t = threading.Thread(target=capture_frames)
t.daemon = True
t.start()

# Start the servo worker thread
servo_thread = threading.Thread(target=servo_worker)
servo_thread.daemon = True
servo_thread.start()

if __name__ == '__main__':
    # Use Flask development server only for local testing
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
