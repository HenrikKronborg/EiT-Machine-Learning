#!/usr/bin/python -i

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters -------------------------------------------------------------

"""
If the difference between image height/width and height/width is odd, an extra row will be added to the right and/or bottom side.
"""

MAX_HEIGHT = 100
MAX_WIDTH  = 100


# --- Image scaler ------------------------------------------------------------
def scale(image, antialiasing=True):
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
    if antialiasing:
        return image.resize((new_width, new_height), Image.ANTIALIAS)
    else:
        return image.resize((new_width, new_height))
    

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

#------
from PIL import Image, ImageChops, ImageStat #REMOVE?
from functools import reduce
from itertools import accumulate

def trim(image):
    """Trims whitespace around the image
    
    Parameters
    ----------
    image : PIL.ImageFile
        The image that should be cropped
    
    Returns
    -------
    : PIL.ImageFile
        The cropped image
    """
    #https://stackoverflow.com/questions/10615901/
    # Create the background from the top-left pixel
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    
    # Get the difference between the background and the image
    diff = ImageChops.difference(image, bg)
    
    # Take care of noise (magic)
    f = sharpness(image)/2
    diff = ImageChops.add(diff, diff, f, -100)
    
    # Get the boundary box
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    
    # Return the trimmed image
    return image

def sharpness(image):
    """Measures the image sharpness
    
    Parameters
    ----------
    image : PIL.ImageFile
        The image that will be measured
    
    Returns
    -------
    : float
        A value measuring the image sharpness
    """
    #https://stackoverflow.com/questions/6646371/
    array = np.asarray(image, dtype=np.int32)

    gy, gx = np.gradient(array)
    gradient_norm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gradient_norm)
    
    return sharpness

def equalize(image):
    """Equalizes the histogram
    
    Parameters
    ----------
    image : PIL.ImageFile
        The image that will be equalized
    
    Returns
    -------
    : PIL.ImageFile
        The equalized image
    """
    #https://stackoverflow.com/questions/7116113/
    #http://effbot.org/zone/pil-histogram-equalization.htm
    histogram = image.histogram()
    
    # len(histogram) is 256 for the dataset
    # step size
    step = reduce(lambda x, y: x+y, histogram)/len(histogram)

    # create equalization lookup table
    lookup_table = [n/step for n in accumulate(histogram)]

    # map image through lookup table
    return image.point(lookup_table)

#-------
import cv2

def crop(image):
    arr = np.asarray(image).copy()

    gray = cv2.blur(arr, (10, 10))
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)


    x, y, w, h = cv2.boundingRect(c)
    # Padding
    dx = 10
    dy = 10
    cv_img = arr[max(0, y-dy):y+h+dy, max(0, x+dx):x+w+dx]

    return Image.fromarray(cv_img)


def ARRAY_FROM_PATH(path):
    """
    Gets the image from the path,
    resizes the image no larger than {MAX_WIDTH}x{MAX_HEIGHT} while keeping aspect ratio,
    makes an array from the image
    pads zeros to make the array {MAX_HEIGHT}x{MAX_WIDTH} (reverse of image size)
    """
    img = Image.open(path)
    img = trim(img)
    img = equalize(img)
    img = crop(img)
    img = scale(img)
    img = equalize(img)
    array_image = np.asarray(img)
    padded_array_image = pad(array_image)
    
    # plt.imshow(img)
    # plt.show()
    
    return padded_array_image
