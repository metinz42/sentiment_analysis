import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")


# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("video.mp4")

face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_DUPLEX

bounding_box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()



while cam.isOpened(): 
    num = 5
    _,frame = cam.read()

    faces = face_detection.detectMultiScale(frame,1.2,10)
    

    if (len(faces)==0):
        cv2.putText(frame,"Yuz Tespit edilemedi!",(100,100),font,1,(0,0,255),2)
    
    else:
        cv2.rectangle(frame,(0,20),(260,130),(0,0,0),-1)
        

    for x,y,w,h in faces:
        cv2.putText(frame,f"{len(faces)} tane yuz tespit edildi!",(0,50),font,0.6,(255,0,0),1)

    if num % 5 == 0 :
        results = model(frame,conf=0.25,agnostic_nms = False)[0]
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                confidence = box.conf[0]  # Doğruluk oranı
                class_id = int(box.cls[0])  # Sınıf kimliği
                label = model.names[class_id]  # Sınıf adı
                confidence_text = f"{confidence:.2f}"
        
        else:
            # Tespit yoksa confidence_text yazdırma
            confidence_text = ""
            label = ""


        detections = sv.Detections.from_ultralytics(results)
        annotated_image = bounding_box_annotator.annotate(
                    scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(
                        scene=annotated_image, detections=detections)


        if (len(faces)>0):
            cv2.putText(annotated_image,f"{label} confidence :{confidence_text}",(0,110),font,0.7,(0,255,0),1)
    
    num += 1

    resized_frame = cv2.resize(annotated_image,(640,480))
    
    cv2.imshow("img",resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()
