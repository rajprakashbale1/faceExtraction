# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:19:22 2018

@author: Rajprakash.Bale
"""
import cv2, dlib, os

# Save the pretrained models locally
# You can download the pretrained models freely from internet; Ex : shape_predictor_68_face_landmarks
# is one such model we used in the following program.
model_path = " model path "
predictor_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)


def imageProcess(image):
    img = cv2.imread(image)                             # read the input image
    r = 1500.0 / img.shape[1]                           # resize the image 
    dim = (1500, int(img.shape[0] * r))
    image_cv = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    dets = detector(image_cv,1)
    for k,d in enumerate(dets):
        shape = sp(image_cv, d)
        s = shape.rect
        face_image = image_cv[s.top():(s.top()+s.height()), s.left():(s.left()+s.width())]
        name = str(k) + '.jpg'
        cv2.imwrite(name, face_image)
    return len(dets)
    

if __name__ == '__main__':
    image_path = " image path "
    no_of_faces = imageProcess(image_path)
    print(no_of_faces)