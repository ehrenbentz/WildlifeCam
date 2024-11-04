#!/bin/bash

# Start a new screen session and run birdcam.py
#screen -dmS birdcam bash -c 'python3 birdcam.py'

# Start ngrok
ngrok http 5000 --url wildlifecam.ngrok.io
#ngrok http 5000 --basic-auth "birdcam:Preston1" --url wildlifecam.ngrok.io
