from PIL import Image, ImageDraw
import ImageTools
import math, time, os
import cv2, numpy as np
from datetime import datetime
import copy

FACE = "haarcascade_frontalface_default.xml"
EYES = "haarcascade_eye.xml"
FILTER = "dogface.png"
DEVICE_ID = 0
FLIP = False

def paste_with_alpha(larger, smaller, xy):
    x_offset, y_offset = xy
    # Stolen from https://stackoverflow.com/a/14102014
    y1, y2 = y_offset, y_offset + smaller.shape[0]
    x1, x2 = x_offset, x_offset + smaller.shape[1]
    alpha_s = smaller[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        larger[y1:y2, x1:x2, c] = (alpha_s * smaller[:, :, c] +
                                alpha_l * larger[y1:y2, x1:x2, c])
    return larger

camera = cv2.VideoCapture(DEVICE_ID)
face_detector = cv2.CascadeClassifier(FACE)
filter_image_orig = Image.open(FILTER)
filter_image = cv2.cvtColor(np.array(filter_image_orig), cv2.COLOR_RGB2BGRA)
width  = int(camera.get(3))
height = int(camera.get(4))
pause = False

ret, full_photo = camera.read()

while True:
    if not pause:
        ret, full_photo = camera.read()
    if FLIP:
        full_photo = cv2.flip(full_photo, -1)
    orig_photo = copy.deepcopy(full_photo)
    # Create monochrome version of image
    full_grey = cv2.cvtColor(full_photo, cv2.COLOR_BGR2GRAY)
    # Detect any faces
    faces = face_detector.detectMultiScale(full_grey, scaleFactor=1.2, minNeighbors=5, minSize=(100,100))
    for face in faces:
        # Get coordinates of the face
        x,y,w,h = face
        # Create crops of the face
        face_photo = full_photo[y:y+h, x:x+w]
        face_grey = full_grey[y:y+h, x:x+w]
        # Resize the filter image
        filter_image_resized = cv2.resize(filter_image, (w,h), interpolation = cv2.INTER_AREA)
        # Paste the cartoon filter over the image
        full_photo = paste_with_alpha(full_photo, filter_image_resized, (x,y))
    # Show final image
    cv2.imshow("Instalife", full_photo)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # ESC key
        break
    elif k == 32: # Space bar
        if not os.path.exists("orig"):
            os.mkdir("orig")
        if not os.path.exists("renders"):
            os.mkdir("renders")
        filename_partial = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cv2.imwrite("renders/"+filename_partial+".png", full_photo)
        cv2.imwrite("orig/"+filename_partial+".png", orig_photo)
        pause = not pause
print("Done!")

