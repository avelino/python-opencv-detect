#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    test

    sample opencv and detect class

    :copyright: (c) 2011 by the Avelino Labs Team
    :author: Thiago Avelino
    :license: New BSD License
    :version: 0.1
"""



import pygame
import Image
from pygame.locals import *
import sys

import opencv
from opencv import highgui
from opencv.highgui import *
from opencv.cv import *

camera = highgui.cvCreateCameraCapture(0)
def detect_hand(image):
    image_size = cvGetSize(image)

    # create grayscale version
    grayscale = cvCreateImage(image_size, 8, 1)
    cvCvtColor(image, grayscale, CV_BGR2GRAY)

    # create storage
    storage = cvCreateMemStorage(0)
    cvClearMemStorage(storage)

    # equalize histogram
    cvEqualizeHist(grayscale, grayscale)

    cashand = cvLoadHaarClassifierCascade('haarcascade_hand.xml', cvSize(1,1))
    hands = cvHaarDetectObjects(grayscale,
            cashand,
            storage,
            1.1,
            2,
            CV_HAAR_DO_CANNY_PRUNING,
            cvSize(50, 50))

    if hands.total > 0:
        print '=> hand detected!'
        for i in hands:
            cvRectangle(image, cvPoint( int(i.x), int(i.y)),
                          cvPoint(int(i.x + i.width), int(i.y + i.height)),
                          CV_RGB(0, 0, 255), 1, 8, 0)


def detect_eye(image):
    image_size = cvGetSize(image)

    # create grayscale version
    grayscale = cvCreateImage(image_size, 8, 1)
    cvCvtColor(image, grayscale, CV_BGR2GRAY)

    # create storage
    storage = cvCreateMemStorage(0)
    cvClearMemStorage(storage)

    # equalize histogram
    cvEqualizeHist(grayscale, grayscale)

    caseye = cvLoadHaarClassifierCascade('haarcascade_eye.xml', cvSize(1,1))
    eyes = cvHaarDetectObjects(grayscale,
            caseye,
            storage,
            1.1,
            2,
            CV_HAAR_DO_CANNY_PRUNING,
            cvSize(50, 50))

    if eyes.total > 0:
        print '=> eye detected!'
        for i in eyes:
            cvRectangle(image, cvPoint( int(i.x), int(i.y)),
                          cvPoint(int(i.x + i.width), int(i.y + i.height)),
                          CV_RGB(255, 0, 0), 1, 8, 0)


def detect_face(image):
    image_size = cvGetSize(image)

    # create grayscale version
    grayscale = cvCreateImage(image_size, 8, 1)
    cvCvtColor(image, grayscale, CV_BGR2GRAY)

    # create storage
    storage = cvCreateMemStorage(0)
    cvClearMemStorage(storage)

    # equalize histogram
    cvEqualizeHist(grayscale, grayscale)

    # detect objects
    cascade = cvLoadHaarClassifierCascade('haarcascade_frontalface_alt.xml', cvSize(1,1))
    faces = cvHaarDetectObjects(grayscale,
            cascade,
            storage,
            1.2,
            2,
            CV_HAAR_DO_CANNY_PRUNING,
            cvSize(50, 50))

    if faces.total > 0:
        print '=> face detected!'
        for i in faces:
            cvRectangle(image, cvPoint( int(i.x), int(i.y)),
                          cvPoint(int(i.x + i.width), int(i.y + i.height)),
                          CV_RGB(0, 255, 0), 2, 8, 0)


def get_image():
    im = highgui.cvQueryFrame(camera)
    detect_face(im)
    detect_eye(im)
    detect_hand(im)
    return opencv.adaptors.Ipl2PIL(im)

fps = 30.0
pygame.init()
window = pygame.display.set_mode((640,480))
pygame.display.set_caption("WebCam Demo")
screen = pygame.display.get_surface()

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == QUIT or event.type == KEYDOWN:
            sys.exit(0)

    im = get_image()
    pg_img = pygame.image.frombuffer(im.tostring(), im.size, im.mode)
    screen.blit(pg_img, (0,0))
    pygame.display.flip()
    pygame.time.delay(int(1000 * 1.0/fps))
