from PIL import Image, ImageDraw
import ImageTools
import math, time
import cv2, numpy as np
from vcam import vcam,meshGen # https://www.learnopencv.com/funny-mirrors-using-opencv/


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
eye_detector = cv2.CascadeClassifier(EYES)
filter_image_orig = Image.open(FILTER)
filter_image = cv2.cvtColor(np.array(filter_image_orig), cv2.COLOR_RGB2BGRA)
width  = int(camera.get(3))
height = int(camera.get(4))

### VIRTUAL CAMERA SETUP
# https://www.learnopencv.com/funny-mirrors-using-opencv/
# Create a virtual camera object. Here H,W correspond to height and width of the input image frame.
c1 = vcam(H=height,W=width)
# Create surface object
plane = meshGen(height,width)
# We generate a mirror where for each 3D point, its Z coordinate is defined as Z = 20*exp^((x/w)^2 / 2*0.1*sqrt(2*pi))
plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
# Get modified 3D points of the surface
pts3d = plane.getPlane()
# Project the 3D points and get corresponding 2D image coordinates using our virtual camera object c1
pts2d = c1.project(pts3d)

while True:
    ret, full_photo = camera.read()
    if FLIP:
        full_photo = cv2.flip(photo, -1)
    # Create monochrome version of image
    full_grey = cv2.cvtColor(full_photo, cv2.COLOR_BGR2GRAY)
    # Detect any faces
    faces = face_detector.detectMultiScale(full_grey, scaleFactor=1.2, minNeighbors=5, minSize=(300,300))
    for face in faces:
        # Get coordinates of the face
        x,y,w,h = face
        # Create crops of the face
        face_photo = full_photo[y:y+h, x:x+w]
        face_grey = full_grey[y:y+h, x:x+w]
        # Detect any eyes in the face
        """
        eyes = eye_detector.detectMultiScale(face_grey, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
        for eye in eyes:
            ex, ey, ew, eh = eye
            ex += x
            ey += y
            cv2.rectangle(full_photo, (ex,ey), (ex+ew,ey+eh), (255,0,0), 2)
        """
        # Resize the filter image
        filter_image_resized = cv2.resize(filter_image, (w,h), interpolation = cv2.INTER_AREA)
        # Paste the cartoon filter over the image
        full_photo = paste_with_alpha(full_photo, filter_image_resized, (x,y))
    # Apply image distortion
    img = full_photo
    map_x,map_y = c1.getMaps(pts2d)
    output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)
    full_photo = output
    # Show final image
    cv2.imshow("Instalife", full_photo)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # ESC key
        break
print("Done!")

