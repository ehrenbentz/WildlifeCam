#! /usr/bin/bash

# Check if rclone is already running
if pgrep -x "rclone" > /dev/null
then
    echo "rclone is already running. Exiting."
else
    echo "Starting rclone upload..."
    rclone copy /media/pi/WLCam GoogleDrive:WildlifeCamRecordings --bwlimit 10M
fi
