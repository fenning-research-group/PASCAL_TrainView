from frgtrainview.analysis.crop import crop_pl
import numpy as np

def test_crop_is_square():
    """
    Since the wafer is a square, the cropped image's pre-transformed dimensions should also be a square.
    If not, then the test fails.
    """
    
    distance_tolerance = 0.10 # acceptable percent difference in width vs height before test fails
    angle_tolerance = 10 # acceptable degree difference from 90 degrees before test fails

    # compute side lengths and angles from corners of the image
    # 0 = top left, 1 = top right, 2 = bottom left, 3 = bottom right
    corners = []

    top_side = corners[1] - corners[0]
    left_side = corners[2] - corners[0]
    bottom_side = corners[3] - corners[2]
    right_side = corners[3] - corners[1]

    # first assert angles are close to 90 degrees
    get_cos = lambda a, b : np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) # helper function to get the cosine between two vectors
    is_valid_angle = lambda theta : np.abs(theta - 90) < angle_tolerance # helper function to ensure angle is close to 90 degrees
    
    cos_angle_top_left = get_cos(top_side, left_side)
    assert is_valid_angle(np.degrees(np.arccos(cos_angle_top_left))), 'Top left is not close to 90 degrees'

    cos_angle_bottom_right = get_cos(bottom_side, right_side)
    assert is_valid_angle(np.degrees(np.arccos(cos_angle_bottom_right))), 'Bottom right is not close to 90 degrees'
    
    cos_angle_top_right = get_cos(top_side, right_side)
    assert is_valid_angle(np.degrees(np.arccos(cos_angle_top_right))), 'Top right is not close to 90 degrees'

    cos_angle_bottom_left = get_cos(bottom_side, left_side)
    assert is_valid_angle(np.degrees(np.arccos(cos_angle_bottom_left))), 'Bottom left is not close to 90 degrees'

    # now assert that width and length are close enough
    is_similar_magnitude = lambda a, b: np.linalg.norm(a) / np.linalg.norm(b) < distance_tolerance

    assert is_similar_magnitude(top_side, left_side), 'Width and length are too different in magnitude'
