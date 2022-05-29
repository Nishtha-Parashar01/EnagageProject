from sre_constants import SUCCESS
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "TestImages"
images = []
classNames=[]
myList = os.listdir(path)
print(myList)

for cls in myList:
    currentImg= cv2.imread(f'{path}/{cls}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


def find(images):
    encodelist=[]
    for i in images:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(i)[0]
        encodelist.append(encode)
    return encodelist


def MarkAttendance(name):
    with open ('Attendance.csv', 'r+') as f:
        mydata= f.readlines()
        namelist=[]
        for lines in mydata:
            entry=lines.split(',')
            namelist.append(entry[0])

            if name not in namelist:
                now= datetime.now()
                DTString= now.strftime('%H:%M:%S')
                f.writelines(f'\n {name}, {DTString}')

        

MarkAttendance('Elon')

encodelistknown = find(images) 
print("Encoding Completed")

cam= cv2.VideoCapture(0)
while True:
    SUCCESS, i = cam.read()
    Smallimg= cv2.resize(i,(0,0), None, 0.25,0.25)
    Smallimg=cv2.cvtColor(Smallimg,cv2.COLOR_BGR2RGB)

    Currentframefaces= face_recognition.face_locations(Smallimg)
    encodeCurrentFrame=face_recognition.face_encodings(Smallimg, Currentframefaces)


    for encodeface, faceloc in zip(encodeCurrentFrame, Currentframefaces):
        match = face_recognition.compare_faces(encodelistknown, encodeface)
        faceDist = face_recognition.face_distance(encodelistknown, encodeface)
        #print(faceDist)
        matchIndex=np.argmin(faceDist)

        if match[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(i, (x1,y1), (x2,y2),(0,255,0),2)
            cv2.rectangle(i, (x1,y2-35), (x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(i, name,(x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
            MarkAttendance(name)

            
    cv2.imshow('Webcam', i)
    cv2.waitKey(1)






# faceloc= face_recognition.face_locations(img1)[0]
# encodeElon=face_recognition.face_encodings(img1)[0]
# cv2.rectangle(img1,(faceloc[3],faceloc[0]), (faceloc[1],faceloc[2]),(255,0,255),2 )

# faceloc2= face_recognition.face_locations(img2)[0]
# encodeTest=face_recognition.face_encodings(img2)[0]
# cv2.rectangle(img2,(faceloc2[3],faceloc2[0]), (faceloc2[1],faceloc2[2]),(255,0,255),2 )

# result=face_recognition.compare_faces([encodeElon], encodeTest) 
# faceDist=face_recognition.face_distance([encodeElon], encodeTest)

