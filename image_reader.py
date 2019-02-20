#!/usr/bin/python -i

from PIL import Image
import numpy as np

# --- Parameters -------------------------------------------------------------

"""
If the difference between image height/width and height/width is odd, an extra row will be added to the right and/or bottom side.
"""

MAX_HEIGHT = 300
MAX_WIDTH  = 300


# --- Image scaler ------------------------------------------------------------
def scale(image):
    """
    This scales the image such that it is not taller nor wider than the maximum width/height given, while keeping the aspect ratio.
    """
    # Get image size
    image_width, image_height = image.size
    
    # Get scaling constant
    scale = min(MAX_HEIGHT/image_height, MAX_WIDTH/image_width)
    
    # Compute new width and height
    new_width, new_height = map(lambda e:int(scale*e),
                                (image_width, image_height))
    
    # Return scaled image
    return image.resize((new_width, new_height), Image.ANTIALIAS)
    

# --- Array padder ------------------------------------------------------------
def pad(array):
    """
    This ensures the array is of the (MAX_HEIGHT, MAX_WIDTH) size
    """
    # Get the initial array size
    array_height, array_width = array.shape

    # Compute the top and bottom padding
    missing_height = (MAX_HEIGHT - array_height)
    padded_height_top    = missing_height//2
    padded_height_bottom = missing_height//2 + missing_height%2

    # Compute the left and right padding
    missing_width = (MAX_WIDTH - array_width)
    padded_width_left    = missing_width//2
    padded_width_right   = missing_width//2 + missing_width%2

    # padding tuple-tuple
    padding = (
        (padded_height_top, padded_height_bottom),
        (padded_width_left, padded_width_right)
    )

    # Return the padded image array
    return np.pad(
        array,
        pad_width=padding,
        mode="constant",      # pad a constant value
        constant_values=0,    # set the constant value to zero
    )

def ARRAY_FROM_PATH(path):
    """
    Gets the image from the path,
    resizes the image no larger than {MAX_WIDTH}x{MAX_HEIGHT} while keeping aspect ratio,
    makes an array from the image
    pads zeros to make the array {MAX_HEIGHT}x{MAX_WIDTH} (reverse of image size)
    """
    raw_image = Image.open(path)
    resized_image = scale(raw_image)
    array_image = np.asarray(resized_image)
    padded_array_image = pad(array_image)
    
    return padded_array_image