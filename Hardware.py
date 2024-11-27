import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import RPi.GPIO as GPIO
import time

# Pin assignments (your existing setup)
RED_LED_PIN = 2
GREEN_LED_PIN = 3
IR_SENSOR_PIN = 14
SERVO_PIN = 17  # GPIO 17, physical pin 11

# Set up GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(IR_SENSOR_PIN, GPIO.IN)
GPIO.setup(SERVO_PIN, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

model = load_model('vgg16_model.h5')  

def capture_image():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read image from webcam.")
        return None

    cv2.imwrite('captured_image.jpg', frame)
    print("Image captured and saved as 'captured_image.jpg'.")
    return 'captured_image.jpg'

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    return predicted_class

def move_servo(prediction):
    if prediction == 0:
        print("Biodegradable detected. Rotating servo to +60 degrees.")
        servo.ChangeDutyCycle(3)  
    else:
        print("Non-biodegradable detected. Rotating servo to -60 degrees.")
        servo.ChangeDutyCycle(12)  

    time.sleep(1)  # Keep the servo at the angle for 1 second

    print("Resetting servo to neutral position (90 degrees).")
    servo.ChangeDutyCycle(7.5)  
    time.sleep(1)  

    servo.ChangeDutyCycle(0)

# Main loop
try:
    while True:
        if GPIO.input(IR_SENSOR_PIN):
            print("Object far away.")
            GPIO.output(RED_LED_PIN, GPIO.LOW)
            GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
        else:  # Object near
            print("Object detected! Saving image.")
            GPIO.output(RED_LED_PIN, GPIO.HIGH)
            GPIO.output(GREEN_LED_PIN, GPIO.LOW)

            # Capture image and predict
            img_path = capture_image()
            if img_path:
                result = predict_image(img_path)
                move_servo(result)
            else:
                print("Failed to capture image.")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    servo.stop()
    GPIO.cleanup()

