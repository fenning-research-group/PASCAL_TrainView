import numpy as np
import cv2

'''
Defect Metrics
'''

def defect_p(img):
    '''
    Parameters:
    - img: a 3-channel RGB image

    Returns:
    - a tuple of floats, representing the proportion of the image area taken up by each defect (line, splotches)
    - a tuple of numpy arrays, representing the image run through defect detection algorithm (line, splotches)
    '''

    # if image channels were in float format [0,1], then
    # convert image channels to uint8 format [0,255]
    # manual scaling seems better than cv2.normalize function, which might map a float <1 to 255
    # which seems to produce line artificts
    if np.issubdtype(img.dtype, np.floating):
        img = np.uint8(255*img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert brightfield image to grayscale

    # apply canny edge detection
    edges = cv2.Canny(gray, 40, 60, apertureSize=3)
    edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1) # dilate to expand the edges and connect lines
    # now find the contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = np.zeros(edges_dilated.shape, np.uint8)
    cont_img = cv2.drawContours(cont_img, contours, -1, color=255, thickness=3)

    # now perform a floodfill to separate contour exterior from interior
    # anything "outside" a defect is connected, and anything "inside" a defect is connected
    
    # first add padding to outside so that any lines that bisect the image don't mess up the floodfill
    b = 1 # border padding amount
    cont_img = cv2.copyMakeBorder(cont_img, b, b, b, b, cv2.BORDER_CONSTANT, 0)
    h, w = cont_img.shape
    mask = np.zeros((h+2, w+2), np.uint8) # floodfill mask
    cv2.floodFill(cont_img, mask, (0,0), 123) # fill with grey so it doesn't obscure edges
    cont_img = cv2.inRange(cont_img, 122, 124) # threshold so that anything outside a defect
    cont_img = cv2.bitwise_not(cont_img) # now invert colors so that contour interior is white and exterior is black
    cont_img = cont_img[b:-b, b:-b] # now crop to remove border padding

    # now remove lines from the contour image so we're just left with splotches and circles
    remove_lines_img = line_defect_p(cv2.cvtColor(cont_img, cv2.COLOR_GRAY2RGB))[1]
    line_img_dilate = cv2.dilate(remove_lines_img, np.ones((5,5)), iterations=1) # dilate lines so that we're sure we remove them
    cont_img = np.maximum(0, cont_img - line_img_dilate) # subtract out lines and rectify
    cont_img = cv2.erode(cont_img, np.ones((5,5)), iterations=1) # erode contours to remove any leftover lines
    cont_img = cv2.dilate(cont_img, np.ones((5,5)), iterations=1) # now get back the area in the real contours that was eroded

    # now remove splotches from the line image so it only has lines
    line_img = line_defect_p(img)[1]
    cont_img_dilate = cv2.dilate(cont_img, np.ones((5,5)), iterations = 1)
    line_img = np.maximum(0, line_img - cont_img_dilate) # subtract out splotches and rectify

    # compute proportion of image taken up by defects
    cont_p = cont_img.sum()/cont_img.size/255
    line_p = line_img.sum()/line_img.size/255

    return (line_p, cont_p), (line_img, cont_img)


def splotch_defect_p(img):
    '''
    Parameters:
    - img: a 3-channel RGB image

    Returns:
    - a float, representing the proportion of the image area taken up by splotches
    - a numpy array, representing the image run through splotch detection algorithm
    '''

    # if image channels were in float format [0,1], then
    # convert image channels to uint8 format [0,255]
    # manual scaling seems better than cv2.normalize function, which might map a float <1 to 255
    # which seems to produce line artificts
    if np.issubdtype(img.dtype, np.floating):
        img = np.uint8(255*img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert brightfield image to grayscale

    # apply canny edge detection
    edges = cv2.Canny(gray, 40, 60, apertureSize=3)
    edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1) # dilate to expand the edges and connect lines
    # now find the contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = np.zeros(edges_dilated.shape, np.uint8)
    cont_img = cv2.drawContours(cont_img, contours, -1, color=255, thickness=3)

    # now perform a floodfill to separate contour exterior from interior
    # anything "outside" a defect is connected, and anything "inside" a defect is connected
    
    # first add padding to outside so that any lines that bisect the image don't mess up the floodfill
    b = 1 # border padding amount
    cont_img = cv2.copyMakeBorder(cont_img, b, b, b, b, cv2.BORDER_CONSTANT, 0)
    h, w = cont_img.shape
    mask = np.zeros((h+2, w+2), np.uint8) # floodfill mask
    cv2.floodFill(cont_img, mask, (0,0), 123) # fill with grey so it doesn't obscure edges
    cont_img = cv2.inRange(cont_img, 122, 124) # threshold so that anything outside a defect
    cont_img = cv2.bitwise_not(cont_img) # now invert colors so that contour interior is white and exterior is black
    cont_img = cont_img[b:-b, b:-b] # now crop to remove border padding

    # now remove lines so we're just left with splotches and circles
    line_img = line_defect_p(cv2.cvtColor(cont_img, cv2.COLOR_GRAY2RGB))[1]
    line_img = cv2.dilate(line_img, np.ones((5,5)), iterations=1) # dilate lines so that we're sure we remove them
    cont_img = np.maximum(np.zeros_like(cont_img), cont_img - line_img) # subtract out lines and rectify
    cont_img = cv2.erode(cont_img, np.ones((5,5)), iterations=1) # erode contours to remove any leftover lines
    cont_img = cv2.dilate(cont_img, np.ones((5,5)), iterations=1) # now get back the area in the real contours that was eroded

    return cont_img.sum()/cont_img.size/255, cont_img


def line_defect_p(img):
    '''
    Parameters:
    - img: a 3-channel RGB image

    Returns:
    - a float, representing the proportion of the image area taken up by lines
    - a numpy array, representing the image run through line detection algorithm
    - a list of tuples, representing the lines found
    '''

    # NOTE: this currently does not discriminate between lines and splotches (which are shaped as round-ish fractals)
    # will need to fix up to only select lines

    # if image channels were in float format [0,1], then
    # convert image channels to uint8 format [0,255]
    # manual scaling seems better than cv2.normalize function, which might map a float <1 to 255
    # which seems to produce line artificts
    if np.issubdtype(img.dtype, np.floating):
        img = np.uint8(255*img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert brightfield image to grayscale

    # apply canny edge detection
    edges = cv2.Canny(gray, 40, 60, apertureSize=3)
    edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1) # dilate to expand the edges and connect lines

    # apply hough lines transform to find lines
    # may need to tune resolution and threshold parameters
    minLineLength = edges_dilated.shape[0]*0.03 # 3% of the width of the image
    maxLineGap = 30 # higher means more lines combined together so there are less overall
    lines = cv2.HoughLinesP(edges_dilated, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None:
        lines = np.zeros(0)

    # now visualize the lines
    line_img = np.zeros(gray.shape)
    for l in lines:
        arr = np.array(l[0], dtype=np.int32)
        x1, y1, x2, y2 = arr

        # cv.line draws a line in img from the point(x1,y1) to (x2,y2).
        # 255 denotes the colour of the line to be drawn
        cv2.line(line_img, (x1, y1), (x2, y2), color=255, thickness=2)

    return line_img.sum()/line_img.size/255, line_img, lines
