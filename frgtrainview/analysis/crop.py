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

    # apply a median blur filter to remove noise
    blurred = img
    iterations = 2
    for i in range(iterations):
        blurred = cv2.medianBlur(blurred, 51)

    # now sharpen to amplify edges
    sharpen_kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    # do adaptive thresholding to extract sample
    gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -1)

    # crop image so only section between holders is visible
    # then find contours of the cropped image
    cropped = thresh[minY:maxY, minX:maxX]
    contours, _ = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # now get the bounding box of the largest area square, which is probably the sample
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
    # transform the rectangle back to uncropped coordinates
    biggest_rect[:, 0] += minX
    biggest_rect[:, 1] += minY

    output_shape = (255,255)
    box = np.float32(biggest_rect)
    box = box[[0,1,3,2]] # reorder
    output_pts = np.array([
        [0,0],
        [0, output_shape[1]],
        [output_shape[0], 0],
        [output_shape[0], output_shape[1]],
        ], np.float32)

    M = cv2.getPerspectiveTransform(box, output_pts)
    out = cv2.warpPerspective(img, M, output_shape, flags=cv2.INTER_LINEAR)
    return out, box
