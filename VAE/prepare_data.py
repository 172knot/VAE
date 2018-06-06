import numpy as np
import cv2
import os

path = "/home/dlagroup5/Assignment_3/Fingerprint/Phase2/Lum"
def main():
    for imgs in os.listdir(path):
        print(imgs)

if  __name__=="__main__":
    main()

