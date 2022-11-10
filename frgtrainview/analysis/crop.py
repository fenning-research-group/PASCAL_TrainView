import cv2
import numpy as np

def crop_pl(img, output_shape=None):
    """Crops a PL image so that only the sample is visible.

    Args:
        img (ndarray): a PL image.
        output_shape (Tuple[int, int]): desired size for the output image.
    
    Returns:
        ndarray: a cropped PL image containing only the sample, with shape==output_shape if given.
        Otherwise, the shape is the size of the bounding box that contains the sample in the original image img.
        ndarray: an array of length 4 representing the corners of the bounding box the sample is contained within on the original image img.
    """
    # define the center of the image between the holder and
    minX, maxX = (200, 1440-200)
    minY, maxY = (0, 1080)

    # create a mask representing each of 4 quadrants
    centerX = (maxX + minX)//2
    centerY = (maxY + minY)//2
    q1 = ((minX, minY), (centerX, centerY))
    q2 = ((centerX, minY), (maxX, centerY))
    q3 = ((centerX, centerY), (maxX, maxY))
    q4 = ((minX, centerY), (centerX, maxY))
    q = [q1,q2,q3,q4]

    quadrant_masks = []
    for i in range(4):
        mask = np.zeros(img.shape[:2], np.uint8)
        x1,y1 = q[i][0]
        x2,y2 = q[i][1]
        mask[y1:y2, x1:x2] = 1
        quadrant_masks.append(mask)

    # now grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.inRange(gray, 250, 256) # threshold on white, the border of sample

    # use houghline transform to clean up and create border
    minLineLength = thresh.shape[0]*0.03
    maxLineGap = 200
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None:
        lines = np.zeros(0)

    # now draw the lines
    line_img = np.zeros_like(gray, np.uint8)
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.int32)
        x1, y1, x2, y2 = arr
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 4)

    # now find the corners
    minDistance = 90
    n_corners = 3
    quad_corners = []
    corner_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for i in range(4):
        corners = cv2.goodFeaturesToTrack(line_img, n_corners, 0.10, minDistance=minDistance, mask=quadrant_masks[i])
        corners = np.int0(corners)
        for c in corners:
            x,y = c.ravel()
            cv2.circle(corner_img, (x,y), 25, (255,0,0), -1)
        quad_corners.append(corners)
        x1,y1 = q[i][0]
        x2,y2 = q[i][1]
        cv2.rectangle(corner_img, (x1, y1), (x2, y2), (255,0,0), 20)

    # find bounding box around all the corners found
    temp = []
    for x in quad_corners:
        temp.extend(x)
    rot_rect = cv2.minAreaRect(np.array(temp, np.float32))
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    cv2.drawContours(corner_img, [box], 0, (0,0,255), 10)

    # use rectangle corners to perspective transform
    # [0] = top left, [1] = top right, [2] = bottom right, [3] = bottom left
    if output_shape == None:
        # if output_shape is not provided, then compute it!
        # we'll just assume we want to keep the same overall size
        # this computes the sidelengths of the biggest rectangle
        output_shape = (int(np.linalg.norm(box[1] - box[0])),
                int(np.linalg.norm(box[3] - box[0])))

    box = np.float32(box)
    output_pts = np.array([
        [0,0],
        [output_shape[0], 0],
        [output_shape[0], output_shape[1]],
        [0, output_shape[1]],
        ], np.float32)

    M = cv2.getPerspectiveTransform(box, output_pts)
    out = cv2.warpPerspective(img, M, output_shape, flags=cv2.INTER_LINEAR)
    return out, box
