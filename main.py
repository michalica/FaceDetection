import numpy as np
import cv2
import os, os.path
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib

IMAGE_DIR = "gt_db/s01/"  # specify your path here
DELAY_MILISECONDS = 2000

image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

face_cascade = cv2.CascadeClassifier('/Users/tomasmichalica/PycharmProjects/FaceRecognition/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/tomasmichalica/PycharmProjects/FaceRecognition/venv/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)


# create a list all files in directory and
# append files with a vaild extention to image_path_list
for file in os.listdir(IMAGE_DIR):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(IMAGE_DIR, file))

# loop through image_path_list to open each image
for imagePath in image_path_list:
    image = cv2.imread(imagePath)

    # display the image on screen with imshow()
    # after checking that it loaded
    if image is not None:
        img = cv2.imread(imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = imutils.resize(img, width=800)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)

        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            img = fa.align(img, gray, rect)

            # display the output images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.imshow('img', img)
            cv2.waitKey(DELAY_MILISECONDS)

cv2.destroyAllWindows()
