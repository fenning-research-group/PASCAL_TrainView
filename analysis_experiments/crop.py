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
    """
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

    # close in gaps to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    filled = cv2.morphologyEx(line_img, cv2.MORPH_CLOSE, kernel, iterations=4)
    filled = cv2.bitwise_not(filled)

    # now get the bounding box of the largest area square, which is probably the sample
    contours, _ = cv2.findContours(filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_rect = None
    biggest_area = 0
    for c in contours:
        # get rotated rectangle from contour
        rot_rect = cv2.minAreaRect(c)
        area = cv2.contourArea(c)

        if area > biggest_area:
            box = cv2.boxPoints(rot_rect)
            box = np.int0(box)
            biggest_rect = box
            biggest_area = area
    
    # use rectangle corners to perspective transform
    # [0] = top left, [1] = top right, [2] = bottom right, [3] = bottom left
    if output_shape == None:
        # if output_shape is not provided, then compute it!
        # we'll just assume we want to keep the same overall size
        # this computes the sidelengths of the biggest rectangle
        output_shape = (int(np.linalg.norm(biggest_rect[1] - biggest_rect[0])),
                int(np.linalg.norm(biggest_rect[3] - biggest_rect[0])))

    output_pts = np.array([
        [0,0],
        [output_shape[0], 0],
        [output_shape[0], output_shape[1]],
        [0, output_shape[1]],
        ], np.float32)

    biggest_rect = np.float32(biggest_rect)
    M = cv2.getPerspectiveTransform(biggest_rect, output_pts) # needs float32 arrays
    out = cv2.warpPerspective(img, M, output_shape, flags=cv2.INTER_LINEAR)
    return out
