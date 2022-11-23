from bs4 import BeautifulSoup # to parse svg file
import cv2
import numpy as np
import cairosvg
from PIL import Image
from io import BytesIO
import os

def overlay_mask(img, mask_svg, coords=None):
    # superimpose the device mask on top of the wafer
    # TODO: orient the device mask to fit with scribe
    
    # determine if mask_svg parameter is a file or a svg figure string
    dpi = img.shape[0]*4 # set dpi big enough for good resolution
    img_png = None
    if os.path.isfile(mask_svg):
        # if so, open the file
        with open(mask_svg, 'r') as f:
            img_png = cairosvg.svg2png(file_obj=f, dpi=dpi)
    else:
        # otherwise, try interpreting the text as an svg figure
        img_png = cairosvg.svg2png(mask_svg, dpi=dpi)
    
    # now format the svg
    mask_img = np.asarray(Image.open(BytesIO(img_png)))

    # recolor the mask
    alpha = mask_img[:,:,3]
    hsv = cv2.cvtColor(mask_img[:,:,0:3], cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    s[alpha > 0] = 255
    v[alpha > 0] = 255
    hnew = np.uint8(np.mod(h + 120, 180))
    hsv = cv2.merge([hnew, s, v])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    mask_img = np.concatenate((rgb, np.expand_dims(alpha, axis=2)), axis=2)

    # now superimpose mask on top of original image
    out = None
    if coords is not None:
        # if coords are given, do a perspective transform to place the mask there
        input_shape = mask_img.shape
        input_pts = np.array([
            [0,0],
            [input_shape[0], 0],
            [input_shape[0], input_shape[1]],
            [0, input_shape[1]],
            ], np.float32)
        M = cv2.getPerspectiveTransform(input_pts, coords) # needs float32 arrays
        out = cv2.warpPerspective(mask_img, M, img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)
    else:
        # if no coords are given, then simply draw the mask on top
        mask_img = cv2.resize(mask_img, img.shape[:2])
        out = np.swapaxes(mask_img, 0, 1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img[out[:,:,3] > 0] = out[out[:,:,3] > 0]

    return img

def extract_mask_region(mask_file, id):
    """Selects a specific region of a device mask svg file by id.

    Args:
        mask_file (pathlike): a path to a svg file containing the device mask.
        id (str): the id of the region to extract; this function looks for the element in the svg file with the attribute id="{id}".
    
    Returns:
        str: a string representing a complete svg figure that can be used to draw the region specified by id.
    """
    # create the xml parsing tree
    with open(mask_file) as f:
        doc = BeautifulSoup(f)

    x = doc.find(attrs={'id': id})

    # build up the svg, ignoring everything else besides the direct parents of the id
    svg = x
    while svg.parent.name is not None:
        temp = svg.parent
        temp.string = '' # remove other children
        temp.append(svg)
        svg = temp
        if svg.name == 'svg':
            break
    svg['viewBox'] = svg['viewbox'] # need to capitalize the B or the drawing breaks
    del svg['viewbox']

    return svg.prettify()
