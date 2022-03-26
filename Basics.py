import cv2
import numpy as np
import face_recognition

imgSalah = face_recognition.load_image_file('imageBasic/Mo Salah.jpg')
imgSalah = cv2.cvtColor(imgSalah,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imageBasic/Mo Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSalah)[0]
encodeSalah = face_recognition.face_encodings(imgSalah)[0]
cv2.rectangle(imgSalah,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeSalah], encodeTest)
faceDis = face_recognition.face_distance([encodeSalah], encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Mo Salah', imgSalah)
cv2.imshow('Mo Test', imgTest)
cv2.waitKey()