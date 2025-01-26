#Importing Libraries
import numpy as np
import argparse
import pickle
import cv2
import os
import time
from keras.models import load_model
from collections import deque

#'http://192.168.1.8:8080'

#Function Definition
def print_results(video, limit=None):
    # fig=plt.figure(figsize=(16, 30))
    #Setting Up Output Directory
    if not os.path.exists('output'):
        os.mkdir('output')
        
    #Loading the Model
    print("Loading model ...")
    model = load_model("C:\\mugilan\\rotaract\\modelnew.h5")
    
    #Initializing Video Stream and Variables
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture('http://192.168.43.1:8080/video')
    writer = None
    (W, H) = (None, None)
    count = 0
    flag = 1
    #Processing Frames
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        #End of Video Stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 128x128, and then
        # perform mean subtraction
        
        #Prepare the Frame for Prediction
        output = frame.copy()
        dummy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        # make predictions on the frame and then update the predictions
        # queue
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        #             print("preds",preds)
        Q.append(preds)

        # perform prediction averaging over the current history of
        # previous predictions
        results = np.array(Q).mean(axis=0)
        i = (preds > 0.70)[0]
        print(i)
        label = i
        text_color = (0, 255, 0)  # default : green
        
        #Violence Detection Logic
        if label:  # Violence prob
            print("violence detected")
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

            # read the haarcascade to detect the faces in an image
            #Face Detection
            face_cascade = cv2.CascadeClassifier('C:\\mugilan\\rotaract\haarcascade_frontalface_default.xml')

            # detects faces in the input image
            #Face Detection
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            # print('Number of detected faces:', len(faces))

            # loop over all detected faces
            if len(faces) > 0:
            #Draw Rectangles Around Detected Faces
                for i, (x, y, w, h) in enumerate(faces):
                    # To draw a rectangle in a face
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    face = output[y:y + h, x:x + w]
                    cv2.imshow("Cropped Face", face)
                    cv2.imwrite("C:\\mugilan\\rotaract\\face_output\\image.png", face)



        #If No Violence Detected
        else:
            print("No violence")
            cv2.imwrite("C:\\mugilan\\rotaract\image_output\\image.png",output)

        # check if the video writer is None
        #Write Processed Frames
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output/v_output.avi", fourcc, 30, (W, H), True)

        # write the output frame to disk
        writer.write(output)

        # show the output image
        #Display Output Frame
        cv2.imshow("frame",output)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # release the file pointersq
    #print("[INFO] cleaning up...")
    #Release Resources
    writer.release()
    vs.release()
    #Run the Function
V_path = "C:\\mugilan\\rotaract\\Testing videos\\V_19.mp4"
NV_path = "/nonv.mp4"
print_results(V_path)
