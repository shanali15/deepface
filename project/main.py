from deepface import DeepFace
import pandas as pd
import cv2
faceCascade = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
models = ["OpenFace"]

while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        faces = frame[x:x+w,y:y+h]
        for model in models:
            
            # result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = model)
            try:
                df = DeepFace.find(img_path = frame, db_path = "Data/", model_name = model)
                # print(df)
                # df = DeepFace.find(img_path = "test/Nafeel80.jpg", db_path = "Data/", model_name = model)
                person = (df.iloc[0])
                person = (person['identity'])
                person = person.split("/")
                person = person[0].split("\\")
                # print (person)
                print(df)
            except:
                pass
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imshow('Video1', faces)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()