
import numpy as np
import cv2
import sqlite3

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade =  cv2.CascadeClassifier('haarcascade_eye.xml')

rec = cv2.face.LBPHFaceRecognizer_create();
rec.read('recognizer\\trainData.yml')
cam = cv2.VideoCapture(0);

def getProfile(id):
    conn = sqlite3.connect("Fasebase.db")
    cmd = "SELECT * FROM People WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

id =0;

font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img,"ID : "+str(profile[0]),(x,y+h+40),font,1,(0,0,255))
            cv2.putText(img,"Name: "+str(profile[1]),(x-46,y+h+70),font,1,(255,0,255))
            cv2.putText(img,"Age: "+str(profile[2]),(x-10,y+h+100),font,1,(0,255,255))
            cv2.putText(img,"Gender: "+str(profile[3]),(x-60,y+h+140),font,1,(255,255,0))
        
    cv2.imshow("Face",img);
    cv2.waitKey(1)
    if(cv2.waitKey(1)==ord('q')):
        break;
    
cam.release()
cv2.destroyAllWindows()





































    
        
