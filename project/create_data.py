import cv2
import numpy as np

from deepface.commons import functions
def creates_dataset():
    cap = cv2.VideoCapture(0)
    opencv_path = functions.get_opencv_path()
    face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_detector_path)
    print(face_cascade)
    while True:
        ret,img = cap.read()
        print(img)
        # raw_img = img.copy()
        resolution = img.shape
        resolution_x = img.shape[1]; resolution_y = img.shape[0]
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        if len(faces) == 0:
            face_included_frames = 0
        else:
            faces = []
        # for (x,y,w,h) in faces:
		# 	if w > 130: #discard small detected faces
				
		# 		face_detected = True
		# 		if face_index == 0:
		# 			face_included_frames = face_included_frames + 1 #increase frame for a single face
				
		# 		cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
				
		# 		cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
				
		# 		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        cv2.imshow('img',img)
        print (faces)
        # cv2.imshow("img",faces)
creates_dataset()
