# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:55:38 2022

@author: Shane
"""

import cv2
from mtcnn import MTCNN
from tensorflow import keras
import dlib
import logging
import cvlib as cv




logger = logging.getLogger(__name__) 




class CVLIB_DNN():
    
    def __init__(self):
        pass


    def get_face_locations(self, gray_scale_image):
        return cv.detect_face(gray_scale_image)[0]



class HAAR():
    
    def __init__(self, haar_cascade):
        
        self.face_classifier = cv2.CascadeClassifier(haar_cascade)


    def get_face_locations(self, gray_scale_image, scale_factor = 1.3, min_neighbors = 5):
        return self.face_classifier.detectMultiScale(gray_scale_image, scaleFactor = scale_factor, minNeighbors = min_neighbors)




class MTCNN():
    
    def __init__(self):
        self.face_classifier = MTCNN()
        
    def get_face_locations(self, rgb_image):
        faces = self.face_classifier.detect_faces(rgb_image)                    #Note MTCNN face detection classifier expects RGB image
        face_locations = []
        for face in faces:
            face_locations.append(face['box'])
            
        return face_locations



class YOLO_v3():
    
    def __init__(self, trained_model = '../trained_models/yolo_v3_face_detection_model.h5'):
        self.face_classifier = keras.models.load_model(trained_model, compile=False)

    def get_face_locations(self, image):
        faces = self.face_classifier.predict(image) 
        


class HOG():
    
    def __init__(self):
        self.face_classifier = dlib.get_frontal_face_detector()
        
    def get_face_locations(self, rgb_image):
        rectangles = self.face_classifier(rgb_image)
        
        face_locations = []
        
        for rect in rectangles:
            startX = rect.left()
            startY = rect.top()
            endX = rect.right()
            endY = rect.bottom()
            # ensure the bounding box coordinates fall within the spatial
            # dimensions of the image
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(endX, rgb_image.shape[1])
            endY = min(endY, rgb_image.shape[0])
            # compute the width and height of the bounding box
            w = endX - startX
            h = endY - startY
            # return our bounding box coordinates
            face_locations.append((startX, startY, w, h))
        
        return face_locations

        
        
class MMOD_CNN():
    
    def __init__(self):
        logger.critical('failing in get_face_locations')
        self.face_classifier = dlib.cnn_face_detection_model_v1('model')
    
    
    
    def get_face_locations(self, rgb_image):
        rectangles = self.face_classifier(rgb_image)
        
        
        
        face_locations = []
        
        for rect in rectangles:
            startX = rect.left()
            startY = rect.top()
            endX = rect.right()
            endY = rect.bottom()
            # ensure the bounding box coordinates fall within the spatial
            # dimensions of the image
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(endX, rgb_image.shape[1])
            endY = min(endY, rgb_image.shape[0])
            # compute the width and height of the bounding box
            w = endX - startX
            h = endY - startY
            # return our bounding box coordinates
            face_locations.append((startX, startY, w, h))
        
        return face_locations


