#necessary imports
import numpy as np
import cv2

#accessing webcam and setting the face and eye cascades
#haar cascade classifier algorithm works on line and edge detection or central varying intensity detecttiom

cap = cv2.VideoCapture(0) #creating the video capture object cap
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  #just loads the classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True: 
#starting an infinite loop till we get a keyboard interrupt
    
    ret, frame = cap.read() 
    #ret is a bool value returned by read which shows if frame was read or not
    #if the frame is captured correctly it stores it in frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    #converts face to grayscale
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
    #face detection is done here
    #faces is a rectangle of the form (x,y,w,h)
    #1.3 is the scale factor which is a parameter 
    #specifying how much the image size is reduced at each scale
    #scaling is done to reduce a larger detected object to a smaller one to be detected by the algo
    #scaling more reduces effieciency and increases speed

    #4 is the minNeighbours parameter
    #specifying how many neighbours each candidate should have to retain it
    #higher == less detections (3-6) good values

    for (x, y, w, h) in faces:
       # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]  
        #creating a region of interest in grayscae eye
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1,5) 
        #eye rectangle
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,255,0), 3)
            #created a rectangle around the eye
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:  #ASCII for Esc is 27 so escapes closes the webcam
        break

cap.release()  
#closes webcam
cv2.destroyAllWindows()

#Haar cascade works as a classifier. 
#It classifies positive data points → that are part of our detected object and negative data points → that don’t contain our object.
