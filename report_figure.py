import numpy as np
from PIL import Image, ImageOps
from image_reader import trim, crop, scale, equalize, pad

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

example_image_paths = [
    "boneage-training-dataset/1378.png",
    "boneage-training-dataset/1399.png",
    "boneage-training-dataset/1429.png",
    "boneage-training-dataset/1445.png",
    "boneage-training-dataset/1590.png",
    "boneage-training-dataset/1609.png",
    "boneage-training-dataset/1855.png",
    "boneage-training-dataset/1934.png",
]


example_image_path = example_image_paths[0]

fig, axes = plt.subplots(2, 3)#, figsize=(28, 24))

for row in axes:
    for ax in row:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(aspect='equal')


def fix_ax_ratio(ax, image):
    w, h = image.size
    s = max(image.size)
    dw, dh = map(lambda l: (s-l)//2, (w, h))
    ax.set_xlim(-dw, w+dw) # This breaks imshow
    ax.set_ylim(-dh, h+dh)
    ylim = -dh, h+dh


def image_plot(image, location, background_color=None):
    if background_color: axes[location].set_facecolor(background_color)
    axes[location].imshow(ImageOps.flip(image), cmap='gray')
    fix_ax_ratio(axes[location], image)
    

def arrow_between(location1, location2, side1, side2, broken=False):
    def get_coords(ax, side):
        if side in {'left', 'right'}:
            y = sum(ax.get_ylim())//2
            x = ax.get_xlim()[side=='right']
            
            #fix bug hack
            #x += (side=='left')*sum(ax.get_xlim())/8
        elif side in {'top', 'bottom'}:
            y = ax.get_ylim()[side=='top']
            y -= (side=='top')*sum(ax.get_ylim())/16 #hack
            x = sum(ax.get_xlim())//2
        
        #ax.plot(x,y,'go',markersize=10)
        return x, y
    
    ax1 = axes[location1]
    ax2 = axes[location2]
    
    xy1 = get_coords(ax1, side1)
    xy2 = get_coords(ax2, side2)
    
    
    connectionstyle = "bar,angle=180,fraction=-0.05" if broken else None
    
    p = ConnectionPatch(xyA=xy1, xyB=xy2,
                        coordsA="data", coordsB="data",
                        axesA=ax1, axesB=ax2,
                        connectionstyle=connectionstyle,
                        arrowstyle="-|>",
                        mutation_scale=20, fc="w")
    
    axes[location1].add_artist(p)

graph = "process"

if graph == "process":
    example_image = Image.open(example_image_path)
    image_plot(example_image, (0,0))


    example_image = trim(example_image)
    image_plot(example_image, (0,1))

    arrow_between((0,0), (0,1), 'right', 'left')

    example_image = equalize(example_image)
    image_plot(example_image, (0,2))

    arrow_between((0,1), (0,2), 'right', 'left')

    example_image = crop(example_image)
    image_plot(example_image, (1,0))

    arrow_between((0,2), (1,0), 'bottom', 'top', broken=True)

    example_image = equalize(example_image)
    image_plot(example_image, (1,1))

    arrow_between((1,0), (1,1), 'right', 'left')

    example_image = scale(example_image)
    image_plot(example_image, (1,2))

    arrow_between((1,1), (1,2), 'right', 'left')
    
    plt.savefig("image_processing.pdf", bbox_inches = 'tight',
    pad_inches = 0)

    
if graph == "multi":
    for i, img in enumerate(map(Image.open, example_image_paths)):
        image_plot(img, (0, i))
        img = trim(img)
        img = equalize(img)
        img = crop(img)
        img = equalize(img)
        img = scale(img)
        image_plot(img, (1, i), background_color='black')
        
        arrow_between((0,i), (1,i), 'bottom', 'top')
    
    plt.subplots_adjust(bottom=0.35, top=0.6, hspace=0.1)
    plt.savefig("multi.pdf", bbox_inches = 'tight',
    pad_inches = 0)