import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageChops, ImageStat
from functools import reduce
from itertools import accumulate

if __name__ == "__main__":
    UNPROCESSED_DIR = Path("boneage-training-dataset")
    PROCESSED_DIR = Path(f"corrected-boneage-training-dataset")
    
    TESTING = False

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
        # Return the crop
        return image.crop(bbox)

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
    h = len(histogram)
    
    # step size
    step = reduce(lambda x, y: x+y, histogram)/h

    # create equalization lookup table
    lookup_table = [n/step for n in accumulate(histogram)]

    # map image through lookup table
    return image.point(lookup_table)

if __name__ == "__main__":
    for i, img in enumerate(UNPROCESSED_DIR.iterdir()):
        # get the image
        image = Image.open(img)
        
        # trim the image
        trimmed_image = trim(image)
        
        # equalize the trimmed image
        equalized_image = equalize(trimmed_image)
        
        # save the equalized image
        equalized_image.save(f"PROCESSED_DIR/{img.stem}.png")
        
        
        if not TESTING: continue
        
        # plot the histogram
        plt.plot(equalized_image.histogram())
        plt.show()
        
        # get image sizes
        ow, oh = image.size
        tw, th = trimmed_image.size
        
        # compute trimming ratios
        rw, rh = tw/ow, th/oh

        print(f"Image {i+1}-{img.stem}: Crop W:{rw:.2f}, H:{rh:.2f}. Sharpness O:{s:.2f}.")
        if rw < 0.7 or rh < 0.7 or i%20==0:
            equalized_image.show()
            input("Press any key to continue")

