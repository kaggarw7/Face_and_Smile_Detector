import cv2

# Face and Smile Classifier
face_detector = cv2.CascadeClassifier('C:\Python Project\Face_Smile_Detector\Files\Face.xml')
smile_detector = cv2.CascadeClassifier('C:\Python Project\Face_Smile_Detector\Files\Smile.xml')

webcam = cv2.VideoCapture(0)

while True:

    # Reads the current frame from the webcam
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break
    
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces
    faces = face_detector.detectMultiScale(frame_grayscale, scaleFactor = 1.8)

    for (x, y, w, h) in faces:

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        the_face = frame[y:y+h, x:x+w]

        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detecting Smile 
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 21)
        
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale = 3,
            fontFace = cv2.FONT_HERSHEY_PLAIN, color = (0,0,255))

    cv2.imshow('Smile Detector', frame)

    #Don't autoclose (Wait here in the code and listen for a key press)
    key = cv2.waitKey(10)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# clean up 
webcam.release()
cv2.destroyAllWindows()