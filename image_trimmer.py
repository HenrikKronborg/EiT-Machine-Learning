from PIL import Image, ImageChops
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
    
def change_contrast(img, level):
    # https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)
    
if __name__ == "__main__":
    from pathlib import Path
    names = [2086] #[1952, 1974, 2007, 2058, 2086]
    path = Path("boneage-test-dataset")
    for name in names:
        image = Image.open(path / f"{name}.png")
        trimmed_image = trim(image, 1.0)
        #trimmed_image.show()
    
    for i, img in enumerate(path.iterdir()):
        image = Image.open(img)
        f = sharpness(image)/2
        trimmed_image = trim(image, f)
        F = sharpness(trimmed_image)
        print(F)
        contrast_image = change_contrast(trimmed_image, F*50)
        
        ow, oh = image.size
        tw, th = trimmed_image.size
        rw, rh = tw/ow, th/oh
        
        print(i, rw, rh)
        if rw < 0.7 or rh < 0.7:
            contrast_image.show()
            #input("Press any key to continue")
        
    