#IMPORT LIBRARIES
import sys   #FOR PYTHON VERSION PROCESSES
import os   #FOR PROCESSES OF FOLDERS AND FILES
import cv2   #FOR IMAGE PROCESSING
import numpy as np   #FOR ARRAYS OF IMAGE
import time   #FOR TIME PROCESSES
from playsound import playsound   #FOR PLAYING SOUNDS
from PyQt5.QtWidgets import *   #FOR USER INTERFACE
from keras.models import load_model   #FOR LOADING THE SAVED MODEL
import glob   #FOR PROCESSES OF FOLDERS AND FILES
from keras.preprocessing import image   #FOR PREDICTION OF MODEL
from playsound import playsound   #FOR PLAYING SOUNDS

model = load_model("lbp-relu-adam.h5")   #LOADING MODEL
global result_number   #PREDICTION VALUE OF MODEL
global result_total   #TOTAL OF PREDICTION VALUE
global result_avg   #AVARAGE OF TOTAL PREDICTION VALUE
camera = cv2.VideoCapture(0)   #FOR USING COMPUTER WEBCAM

class Window(QWidget):   #IT CREATES CLASS FOR FORM SCREEN

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.button1 = QPushButton(self)   #CREATE A BUTTON
        self.button1.setText("START")   #SET BUTTON TEXT
        self.button1.resize(200,200)   #SET BUTTON RESIZE
        self.button1.move(575,250)   #SET BUTTON LOCATION IN SCREEN
        self.button1.clicked.connect(self.open_frame)   #BUTTON CLICK METHOD
        self.setWindowTitle("Motion Recognize")   #SET WINDOW TITLE
        self.showMaximized()   #SET WINDOW GEOMETRY

    def open_frame(self):    #BUTTON CLICK EVENTS
        time1 = time.time()   #GIVES THE INSTANT TIME
        while True:   #INFINITE LOOP IS CREATED TO OPENING THE WEBCAM
            ret, frame = camera.read()   #THE IMAGE FROM TAKEN THE WEBCAM IS ASSIGNED TO THE VARIABLE
            crop_image = frame[50:430, 100:540]  # CROPING FRAME FOR RECTANGLE
            frame2 = cv2.resize(crop_image, (224, 224))  # IMAGE RESIZE FOR PREDICTON OF MODEL
            time2 = time.time()      #TAKES THE INSTANT TIME
            diff = int(time2-time1)   #THE DIFFERENCE BETWEEN THE TWO TIMES IS FOUND
            diff2 = 15-diff   #FOR COUNTDOWN
            if cv2.waitKey(20) and diff < 15:   #THE IMAGE FROM TAKEN THE WEBCAM IS SET 20 FPS AND TO KEEP THE DISPLAY ON FOR 15 SECONDS.
                frame = cv2.putText(frame, "PLEASE TAKE YOUR CORRECT TRAINING", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #TEXT IS ADDED ON FRAME
                frame = cv2.putText(frame, "POSITION WITHIN " + str(diff2) + " SECONDS.",(5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   #TEXT IS ADDED ON FRAME
                frame = cv2.putText(frame, str(diff2), (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 0), 6)   #TEXT IS ADDED ON FRAME
                cv2.imshow('CAMERA', frame)   #THE IMAGE IS SHOWN
            else:
                break   #AFTER COUNTDOWN THE SCREEN CLOSES

        result_total = 0   #THE FIRST VALUE IS ASSIGNED FOR RESULT_TOTAL
        result_number = 0   #THE FIRST VALUE IS ASSIGNED FOR RESULT_NUMBER
        result_avg = 0   #THE FIRST VALUE IS ASSIGNED FOR RESULT_AVG
        motion_number = 0   #DEFINE MOTION NUMBER DURING EXERCISE TIME AND THE FIRST VALUE IS ASSIGNED FOR MOTION_NUMBER
        correct_number = 0   #DEFINE CORRECT MOVE NUMBER DURING EXERCISE TIME AND THE FIRST VALUE IS ASSIGNED FOR CORRECT_NUMBER
        correct_percent = 0.0   #DEFINE CORRECT MOVE PERCENTAGE DURING EXERCISE AND THE FIRST VALUE IS ASSIGNED FOR CORRECT_PERCENT
        incorrect_percent = 0.0   #DEFINE INCORRECT MOVE PERCENTAGE DURING EXERCISE AND THE FIRST VALUE IS ASSIGNED FOR INCORRECT_PERCENT

        while True:   #INFINITE LOOP IS CREATED TO OPENING THE WEBCAM
            ret, frame = camera.read()  # THE IMAGE FROM TAKEN THE WEBCAM IS ASSIGNED TO THE VARIABLE
            crop_image = frame[50:430, 100:540]  # CROPING FRAME FOR RECTANGLE
            frame2 = cv2.resize(crop_image, (224, 224))  # IMAGE RESIZE FOR PREDICTON OF MODEL
            time3 = time.time()      #TAKES THE INSTANT TIME
            diff = int(time3 - time1)   #THE DIFFERENCE BETWEEN THE TWO TIMES IS FOUND
            diff2 = 20 - diff   #FOR COUNTDOWN
            cv2.imwrite(os.path.join(r'C:\Users\aydin\Desktop\motion\kamera', 'image.jpg'), frame2)   #PICTURE IS CAPTURED FROM REAL TIME IMAGE FOR PREDICTION
            if cv2.waitKey(20) and diff < 20:   #THE IMAGE FROM TAKEN THE WEBCAM IS SET 20 FPS AND TO KEEP THE DISPLAY ON FOR 5 SECONDS.
                for img in glob.glob(r"C:\Users\aydin\Desktop\motion\kamera\image.jpg"):  # THE PICTURE TAKEN IS ASSIGNED TO THE VARIABLE FOR PREDICTION
                    test_image = image.load_img(img, target_size=(224, 224))  # THE PICTURE IS RESIZED
                    test_image = image.img_to_array(test_image)  # THE PICTURE IS CONVERTED TO ARRAY.
                    test_image = np.expand_dims(test_image, axis=0)  # THE EXPAND_DÝMS() FUNCTION IS USED TO EXPAND THE SHAPE OF AN ARRAY.
                    result = model.predict(test_image)   #THE PREDICTED VALUE THAT THE MODEL DETERMINES FOR THE PICTURE IS ASSIGNED TO A VARIABLE
                    result_total = result_total + int(result[0][0])   #PREDICTION VALUES ARE SUMMED FOR AVERAGE PREDICTION.
                    result_number = result_number + 1 #TO INCREASE result_number
                    frame = cv2.putText(frame, "PLEASE WAIT IN THE CORRECT POSITION",(5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   #TEXT IS ADDED ON FRAME
                    frame = cv2.putText(frame, "WHILE MAKING ADJUSTMENTS.", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   #TEXT IS ADDED ON FRAME
                    frame = cv2.putText(frame, str(diff2), (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 0), 6)   #TEXT IS ADDED ON FRAME
                    cv2.imshow('CAMERA', frame)     # TO SHOW FRAME
            else:
                result_avg = result_total / result_number   # TO GET AVERAGE OF RESULT
                break
        while True:     # INFINITE LOOP IS CREATED TO OPENING THE WEBCAM
            ret, frame = camera.read()       # THE IMAGE FROM TAKEN THE WEBCAM IS ASSIGNED TO THE VARIABLE
            crop_image = frame[50:430, 100:540]     # CROPING FRAME FOR RECTANGLE
            frame2 = cv2.resize(crop_image, (224, 224))     # IMAGE RESIZE FOR PREDICTON OF MODEL
            time4 = time.time()  # TAKES THE INSTANT TIME
            diff = int(time4 - time1) # THE DIFFERENCE BETWEEN THE TWO TIMES IS FOUND
            diff2 = 50 - diff       # FOR COUNTDOWN
cv2.imwrite(os.path.join(r'C:\Users\aydin\Desktop\motion\kamera', 'image.jpg'), frame2) #PICTURE IS CAPTURED FROM REAL TIME IMAGE FOR PREDICTION
            if cv2.waitKey(20) and diff < 50:   #THE IMAGE FROM TAKEN THE WEBCAM IS SET 20 FPS AND TO KEEP THE DISPLAY ON FOR 30 SECONDS.
                for img in glob.glob(r"C:\Users\aydin\Desktop\motion\kamera\image.jpg"):  # THE PICTURE TAKEN IS ASSIGNED TO THE VARIABLE FOR PREDICTION
                    test_image = image.load_img(img, target_size=(224, 224))  # THE PICTURE IS RESIZED
                    test_image = image.img_to_array(test_image)  # THE PICTURE IS CONVERTED TO ARRAY.
                    test_image = np.expand_dims(test_image,axis=0)  # THE EXPAND_DÝMS() FUNCTION IS USED TO EXPAND THE SHAPE OF AN ARRAY.
                    result = model.predict(test_image)  # THE PREDICTED VALUE THAT THE MODEL DETERMINES FOR THE PICTURE IS ASSIGNED TO A VARIABLE
                    if (result_avg*0.998) <= result[0][0] <= (result_avg*1.02):   #A RANGE IS DETERMINED FOR THE ACCURACY OF THE ESTIMATED VALUE
                        prediction = 'CORRECT'   #THE PREDICTION VARIABLE IS ASSIGNED A TEXT
                        frame = cv2.putText(frame, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)   #TEXT IS ADDED ON FRAME
                        frame = cv2.putText(frame, str(diff2), (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)   #TEXT IS ADDED ON FRAME
                        frame = cv2.rectangle(frame, (100, 50), (540, 430), (0, 255, 0),3)   #TEXT IS ADDED ON FRAME
                        cv2.imshow("CAMERA", frame)   #THE IMAGE IS SHOWN
                        motion_number = motion_number + 1 # TO INCREASE motion_number ONE BY ONE
                        correct_number = correct_number + 1  # TO INCREASE correct_number ONE BY ONE
                        playsound('correct.wav')   #THE CORRECT SOUND IS PLAYED
                    else:
                        prediction = 'INCORRECT'   #THE PREDICTION VARIABLE IS ASSIGNED A TEXT
                        frame = cv2.putText(frame, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)   #TEXT IS ADDED ON FRAME
                        frame = cv2.putText(frame, str(diff2), (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)   #TEXT IS ADDED ON FRAME
                        frame = cv2.rectangle(frame, (100, 50), (540, 430), (0, 255, 0), 3)   #TEXT IS ADDED ON FRAME
                        cv2.imshow("CAMERA", frame)  #THE IMAGE IS SHOWN
                        motion_number = motion_number + 1# TO INCREASE motion_number ONE BY ONE
                        playsound('incorrect.mp3')   #THE INCORRECT SOUND IS PLAYED
            else:
                correct_percent = (correct_number / motion_number) * 100   #CORRECT MOTION PERCENTAGE IS CALCULATED
                incorrect_percent = 100 - correct_percent   #INCORRECT MOTION PERCENTAGE IS CALCULATED
                frame = cv2.putText(frame, "Correct Exercise Percentes: %" +str(correct_percent), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)   #TEXT IS ADDED ON FRAME
                frame = cv2.putText(frame, "Incorrect Exercise Percentes: %" +str(incorrect_percent), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)   #TEXT IS ADDED ON FRAME
                frame = cv2.putText(frame, "Exercise Time: 30 Seconds", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)   #TEXT IS ADDED ON FRAME
                cv2.imshow("RESULTS", frame)  #THE IMAGE IS SHOWN
                break
        camera.release()   #WEBCAM IS RELEASED
        cv2.destroyWindow("CAMERA")   #CAMERA FRAME IS DESTROYED.

app = QApplication(sys.argv)   #FORM SCREEN IS DEFINED
menu = Window()   #CREATING WINDOW FOR FORM SCREEN
sys.exit(app.exec_())   #FORM SCREEN IS CLOSED
