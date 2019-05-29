# !/usr/bin/env python

# Import modules
import cv2
import dlib
import numpy as np
import os

# Import dlib face alignment file
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FULL_POINTS = list(range(0, 68))
FACE_POINTS = list(range(17, 68))
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# List selfies while avoiding hidden files
selfies = [x for x in os.listdir('data/selfies') if x[0] != '.']

# Loop over selfies
for selfie in selfies:

    # Read in selfie
    photo = cv2.imread('data/selfies/' + selfie)

    # Convert to gray scale
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    # Detect dlib face rectangles
    factor = 4
    gray = cv2.resize(gray, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_LINEAR)
    rectangles = detector(gray, 0)

    # Track face features if bounding box detected
    if rectangles:

        # Face shape prediction
        shape = predictor(gray, rectangles[0])
        coordinates = np.zeros((shape.num_parts, 2), dtype='int')
        for x in range(0, shape.num_parts):
            coordinates[x] = (shape.part(x).x, shape.part(x).y)
        shape = factor * coordinates

        # Forehead top and side anchors
        forehead_rt = 2 * (shape[19] - shape[36]) + shape[19]
        forehead_lt = 2 * (shape[24] - shape[45]) + shape[24]
        forehead_rs = 2 * (shape[19] - shape[36]) + shape[0]
        forehead_ls = 2 * (shape[24] - shape[45]) + shape[16]

        # Forehead anchor midpoints
        midpoint_r = [0.25 * (forehead_rt[0] - forehead_rs[0]) + forehead_rs[0],
                      0.75 * (forehead_rt[1] - forehead_rs[1]) + forehead_rs[1]]
        midpoint_l = [0.25 * (forehead_lt[0] - forehead_ls[0]) + forehead_ls[0],
                      0.75 * (forehead_lt[1] - forehead_ls[1]) + forehead_ls[1]]

        # Add forehead anchor points
        shape = np.vstack((shape, forehead_rt, forehead_lt, forehead_rs,
                           forehead_ls, midpoint_r, midpoint_l)).astype(np.int)

        # Preallocate mask array
        feature_mask = np.zeros(photo.shape[:2])

        # Generate face mask
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(shape), 1)
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(shape[RIGHT_EYE_POINTS]), 0)
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(shape[LEFT_EYE_POINTS]), 0)
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(shape[MOUTH_OUTLINE_POINTS]), 0)
        photo[~feature_mask.astype(np.bool)] = 0

        # Determine bounding box
        left, top = shape.min(axis=0)
        right, bottom = shape.max(axis=0)
        left = left if left >= 0 else 0
        top = top if top >= 0 else 0
        right = right if right < photo.shape[1] else photo.shape[1] - 1
        bottom = bottom if bottom < photo.shape[0] else photo.shape[0] - 1
        width, height = right - left, bottom - top

        # Extract square bounding box
        if height > width:
            bounding_box = np.zeros((height, height, 3), dtype=np.uint8)
            padding = (height - width) // 2
            bounding_box[:, padding:width + padding] = photo[top:bottom, left:right]
            crop = cv2.resize(bounding_box, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            bounding_box = np.zeros((width, width, 3), dtype=np.uint8)
            padding = (width - height) // 2
            bounding_box[padding:height + padding, :] = photo[top:bottom, left:right]
            crop = cv2.resize(bounding_box, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Write out cropped image
        cv2.imwrite('data/crops/' + selfie.split('.')[0] + '_crop.jpg', crop)
