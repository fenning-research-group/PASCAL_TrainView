from bs4 import BeautifulSoup # to parse svg file

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
