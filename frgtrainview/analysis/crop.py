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
    return _crop_pl_adaptive(img, output_shape=output_shape)

def _crop_pl_adaptive(img, output_shape=None):
    """Helper func. Don't use this directly.
    Uses adaptive thresholding to crop PL image."""
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

    # do morphological ops to clean up noise
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel=np.ones((3,3), np.uint8), iterations=4)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_DILATE, kernel=np.ones((5,5), np.uint8))

    # crop image so only section between holders is visible
    cropped = morphed[minY:maxY, minX:maxX]

    # use houghline transform to clean up and create border
    minLineLength = cropped.shape[0]*0.2
    maxLineGap = 20
    lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None:
        lines = np.zeros(0)

    # now draw the lines
    line_img = np.zeros_like(cropped, np.uint8)
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.int32)
        x1, y1, x2, y2 = arr
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 4)

    # then find contours of the cropped image
    contours, _ = cv2.findContours(line_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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


def _crop_pl_gc(img, output_shape=None):
    """Helper func. Don't use this directly.
    Uses the GrabCut algorithm (https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html) to crop PL image."""
    # set RNG seed for consistency since GrabCut algorithm is random
    cv2.setRNGSeed(0)

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

    # define the center of the image between the holders
    minX, maxX = (200, 1440-200)
    minY, maxY = (0, 1080)

    side_tol = 20
    # define a slice of the image which definitely contains the sample (the center)
    cminX, cmaxX = (int(1440//2 - 310), int(1440//2 + 310))
    cminY, cmaxY = (int(1080//2 - 310), int(1080//2 + 310))

    # define mask for the grabcut
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[minY:maxY, minX:maxX] = cv2.GC_PR_FGD # define the probably foreground (sample is contained within this region)
    mask[:, minX:minX+side_tol] = cv2.GC_PR_BGD
    mask[:, maxX-side_tol:maxX] = cv2.GC_PR_BGD
    mask[cminY:cmaxY, cminX:cmaxX] = cv2.GC_FGD # define the foreground (definitely sample)

    # now do grabcut
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(blurred, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==cv2.GC_PR_BGD)|(mask==cv2.GC_BGD),0,1).astype('uint8')
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=4)
    img_gc = img*mask2[:,:,np.newaxis]

    # find corners using the grabcut result as a mask to improve corner detection
    gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 30, 0.01, 1, mask=mask2).astype(np.int0)
    for c in corners:
        cv2.circle(img_gc, c[0, :], 20, (0,255,255), -1)

    # find bounding box around all the corners found
    if corners.shape[0] > 0:
        rot_rect = cv2.minAreaRect(corners)
    else: # no corners found, can't crop
        rot_rect = None

    # now do perspective transform
    box = cv2.boxPoints(rot_rect)
    box = box[[0,1,3,2]] # reorder
    box = np.float32(box)
    output_pts = np.array([
        [0,0],
        [0, output_shape[1]],
        [output_shape[0], 0],
        [output_shape[0], output_shape[1]],
        ], np.float32)

    M = cv2.getPerspectiveTransform(box, output_pts)
    out = cv2.warpPerspective(img, M, output_shape, flags=cv2.INTER_LINEAR)
    return out, box
