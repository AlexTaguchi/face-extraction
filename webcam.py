# !/usr/bin/env python

# Import modules
import cv2
import dlib
import numpy as np

# Import dlib face alignment file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define facial landmarks
landmarks = {'jawline': list(range(0, 17)),
             'right_eyebrow': list(range(17, 22)),
             'left_eyebrow': list(range(22, 27)),
             'nose': list(range(27, 36)),
             'right_eye': list(range(36, 42)),
             'left_eye': list(range(42, 48)),
             'outer_mouth': list(range(48, 60)),
             'inner_mouth': list(range(60, 68))}

# Face tracking
crop = np.zeros((256, 256, 3), dtype=np.uint8)
videoCapture = cv2.VideoCapture(0)
while True:

    # Read in mirror image video frame
    ret, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)

    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        # Generate face mask
        face_mask = np.zeros(frame.shape[:2])
        cv2.fillConvexPoly(face_mask, cv2.convexHull(shape), 1)
        
        # Overlay focused face over blurred background
        background = cv2.blur(frame, (50, 50))
        background[face_mask.astype(np.bool)] = 0
        frame[~face_mask.astype(np.bool)] = 0
        frame = frame + background

        # Draw landmarks
        for feature, points in landmarks.items():
            if feature == 'nose':
                points += [points[3]]
            elif feature == 'jawline':
                pass
            else:
                points += [points[0]]
            for i in range(len(points) - 1):
                cv2.line(frame, tuple(shape[points[i]]), tuple(shape[points[i+1]]), (0, 0, 255), 2)

        # Determine bounding box dimensions
        left, top = shape.min(axis=0)
        width, height = shape.max(axis=0) - shape.min(axis=0)
        left = left if width >= height else left - int((height - width) / 2)
        top = top if height >= width else top - int((width - height) / 2)
        length = max(width, height)

        # Ensure bounding box is within frame
        if left < 0:
            left = 0
        elif left + length >= frame.shape[1]:
            left -= frame.shape[1] - left - length + 1
        if top < 0:
            top = 0
        elif top + length >= frame.shape[0]:
            top -= frame.shape[0] - top - length + 1

        crop = cv2.resize(frame[top:top + length, left:left + length],
                          (256, 256), interpolation=cv2.INTER_LINEAR)

    # Display the resulting frame
    cv2.imshow('Video', crop)

    # Exit by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, release the capture
videoCapture.release()
cv2.destroyAllWindows()
