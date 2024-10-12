import cv2
import numpy as np
from keras.models import load_model
from picarx import Picarx

from keras.losses import MeanSquaredError

from picamera2 import Picamera2

import time
from time import sleep
import readchar

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.start()

# Load the model, specifying the custom objects if needed
model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})


# Image preprocessing function

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image


# Function to compute steering angle
def compute_steering_angle(frame):
    preprocessed = img_preprocess(frame)
    img = np.asarray([preprocessed])
    steering_angle = model.predict(img)[0]  # Predict the steering angle
    return steering_angle


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    px = Picarx()
    px.set_dir_servo_angle(0)

    while True:
        # Compute the steering angle for the current frame
        img = picam2.capture_array()
        px.forward(0.003)
        steering_angle = compute_steering_angle(img)
        steering_angle = steering_angle[0]
        turn = steering_angle * 50
        px.set_dir_servo_angle(turn)
        print(f"Steering Angle: {turn}")
        # Display the steering angle on the video frame
        cv2.putText(img, f"Steering Angle: {turn:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show the processed video frame
        cv2.imshow('Frame', img)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
    picam2.stop()
    cv2.destroyAllWindows()
    px.stop()