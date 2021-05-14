# @mic34 on repl Original Link: https://replit.com/@mic34/MockSocialSensor#main.py
import cv2
import imutils
import numpy as np
import argparse

def detect(frame):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    person = 1
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_ITALIC, 0.5, (0,0,255), 1)
        person += 1
    
    cv2.putText(frame, f'Number Of People : {person-1}', (40,70), cv2.FONT_ITALIC, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)

    return frame

def detectByPathImage(path, output_path):
    image = cv2.imread(path)

    image = imutils.resize(image, width = min(800, image.shape[1])) 

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def socialSensor(args):
    image_path = args["image"]
    if image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i", "--image", default=None)
    arg_parse.add_argument("-o", "--output", type=str)
    args = vars(arg_parse.parse_args())
    
    return args

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = argsParser()
    socialSensor(args)

