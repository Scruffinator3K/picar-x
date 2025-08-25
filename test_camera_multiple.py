#!/usr/bin/env python3

import cv2

print("Testing camera access...")
cap1 = cv2.VideoCapture(0)
print(f"First capture opened: {cap1.isOpened()}")

cap2 = cv2.VideoCapture(0) 
print(f"Second capture opened: {cap2.isOpened()}")

if cap1.isOpened():
    ret, frame = cap1.read()
    print(f"First capture read: {ret}")

if cap2.isOpened():
    ret, frame = cap2.read()
    print(f"Second capture read: {ret}")

cap1.release()
cap2.release()
