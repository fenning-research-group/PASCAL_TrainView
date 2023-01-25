from crop import crop_pl
import numpy as np
import cv2

def test_crop_is_square():
    """
    Since the wafer is a square, the cropped image's pre-transformed dimensions should also be a square.
    If not, then the test fails.
    """
    
    distance_tolerance = 0.10 # acceptable percent difference in width vs height before test fails
    angle_tolerance = 10 # acceptable degree difference from 90 degrees before test fails

    img = None

    # compute side lengths and angles from corners of the image
    # 0 = top left, 1 = top right, 2 = bottom left, 3 = bottom right
    _, corners = crop_pl(img)
    _is_square(corners, distance_tolerance, angle_tolerance) # all the assertion code in here

    

def _is_square(corners, distance_tolerance:float=0.10, angle_tolerance:float=10):
    """
    Compute side lengths and angles from corners of a shape to determine if is a square

    Parameters:
        corners: a 4-length array with (x,y) coordinates
            0 = top left, 1 = top right, 2 = bottom left, 3 = bottom right
        distance_tolerance (float): the acceptable percent difference in width vs height before the test fails
        angle_tolerance (float): the acceptable degree difference from 90 degrees before the test fails
    """
    top_side = corners[1] - corners[0]
    left_side = corners[2] - corners[0]
    bottom_side = corners[3] - corners[2]
    right_side = corners[3] - corners[1]

    # first assert angles are close to 90 degrees
    get_cos = lambda a, b : np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) # helper function to get the cosine between two vectors
    is_valid_angle = lambda theta : np.abs(theta - 90) < angle_tolerance # helper function to ensure angle is close to 90 degrees
    
    angle_top_left = np.degrees(np.arccos(get_cos(top_side, left_side)))
    assert is_valid_angle(angle_top_left), f'Top left is not close to 90 degrees, is {angle_top_left}'

    angle_bottom_right = np.degrees(np.arccos(get_cos(bottom_side, right_side)))
    assert is_valid_angle(angle_bottom_right), f'Bottom right is not close to 90 degrees, is {angle_bottom_right}'
    
    angle_top_right = np.degrees(np.arccos(get_cos(top_side, right_side)))
    assert is_valid_angle(angle_top_right), f'Top right is not close to 90 degrees, is {angle_top_right}'

    angle_bottom_left = np.degrees(np.arccos(get_cos(bottom_side, left_side)))
    assert is_valid_angle(angle_bottom_left), f'Bottom left is not close to 90 degrees, is {angle_bottom_left}'

    # now assert that width and length are close enough
    is_similar_magnitude = lambda a, b: np.linalg.norm(a) / np.linalg.norm(b) < (1+distance_tolerance)

    assert is_similar_magnitude(top_side, left_side), f'Width and length are too different in magnitude, is {np.linalg.norm(top_side)} vs {np.linalg.norm(left_side)}'


def test_crop_correct_area():
    """
    We know that the based on current magnification, the wafers should take up a specific amount of area of the image.
    If not enough of the original image is cropped out, then the test fails.
    If too much of the image is cropped out, then the test fails.
    Otherwise, the test succeeds.
    """
    # in PL img, the wafer region takes up roughly half of the width and most of the height
    MIN_AREA_PIXELS = 700 * 700
    MAX_AREA_PIXELS = 1080 * 1080 

    img = None
    _, corners = crop_pl(img)
    area = cv2.contourArea(corners)

    assert area > MIN_AREA_PIXELS, f'Crop region is too small, is {area} pixels'
    assert area < MAX_AREA_PIXELS, f'Crop region is too large, is {area} pixels'
