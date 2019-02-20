from PIL import Image, ImageChops, ImageStat
import numpy as np

def trim(im, f):
    #https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, f, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def sharpness(img):
    #https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
    array = np.asarray(img, dtype=np.int32)

    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    
    return sharpness

def sharpness2(img):
    # of poorer quality
    array = np.asarray(img, dtype=np.int32)
    
    dx = np.diff(array)[1:,:] # remove the first row
    dy = np.diff(array, axis=0)[:,1:] # remove the first column
    dnorm = np.sqrt(dx**2 + dy**2)
    sharpness = np.average(dnorm)
    
    return sharpness

def brightness(img):
    stat = ImageStat.Stat(img)
    return stat.mean[0]

def change_contrast(img, level):
    # https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

from functools import reduce
def equalize(im):
    #https://stackoverflow.com/questions/7116113/normalize-histogram-brightness-and-contrast-of-a-set-of-images-using-python-im
    h = im.convert("L").histogram()
    lut = []
    for b in range(0, len(h), 256):
        # step size
        step = reduce(lambda x, y: x+y, h[b:b+256]) / 255
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]
    # map image through lookup table
    return im.point(lut)

if __name__ == "__main__":
    from pathlib import Path
    path = Path("boneage-test-dataset")
    
    for i, img in enumerate(path.iterdir()):
        image = Image.open(img)
        s = sharpness(image)
        trimmed_image = trim(image, s/2)
        b = sharpness(trimmed_image)
        contrast_image = equalize(trimmed_image) #change_contrast(trimmed_image, b*3 0)
        
        ow, oh = image.size
        tw, th = trimmed_image.size
        rw, rh = tw/ow, th/oh
        
        print(f"Image {i+1}-{img.stem}: Crop W:{rw:.2f}, H:{rh:.2f}. Sharpness O:{s:.2f}, T:{b:.2f}.")
        if rw < 0.7 or rh < 0.7 or i%20==0:
            pass
            #contrast_image.show()
            #input("Press any key to continue")
        
        contrast_image.save("corrected-boneage-test-dataset/{img.stem}.png")
        
    