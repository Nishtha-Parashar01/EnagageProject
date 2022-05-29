import cv2
import numpy as np
import face_recognition


img1= face_recognition.load_image_file("C:/Users/Dell/Documents/Dell/Project1/TestImages/ElonMusk.jpg")
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2= face_recognition.load_image_file("C:/Users/Dell/Documents/Dell/Project1/TestImages/ElonTest.jpg")
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

faceloc= face_recognition.face_locations(img1)[0]
encodeElon=face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(faceloc[3],faceloc[0]), (faceloc[1],faceloc[2]),(255,0,255),2 )

faceloc2= face_recognition.face_locations(img2)[0]
encodeTest=face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2,(faceloc2[3],faceloc2[0]), (faceloc2[1],faceloc2[2]),(255,0,255),2 )

result=face_recognition.compare_faces([encodeElon], encodeTest) 
faceDist=face_recognition.face_distance([encodeElon], encodeTest)
print(result, faceDist)
cv2.putText(img2,f'{result} {round(faceDist[0], 2) }', (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255) ,2)

cv2.imshow("Elon Musk", img1)
cv2.imshow("Elon Test", img2)
cv2.waitKey(0)

