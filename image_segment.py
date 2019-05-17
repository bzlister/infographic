import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import label, regionprops
import math
import cv2

def image_show(split_image):
    plt.imshow(split_image)
    plt.show()

image = io.imread('images\\ig1.png')
transparent_px = np.uint8([0,0,0,0])
image_slic = seg.slic(image,n_segments=6) #need to tune this parameter for every image
regions = regionprops(image_slic)
leftover = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2RGBA)
for props in regions:
    pos = props.filled_image
    minr, minc, maxr, maxc = props.bbox
    clip = image[minr:maxr,minc:maxc].copy()
    clip = cv2.cvtColor(clip, cv2.COLOR_RGB2RGBA)
    for ii in range(0, len(clip)):
        for jj in range(0, len(clip[ii])):
            if (pos[ii][jj] == False):
                clip[ii][jj] = transparent_px
            else:
                leftover[ii+minr][jj+minc] = transparent_px
    image_show(clip)

delcount =0
l = len(leftover)
r = 0
while (r < l):
    trans = True
    for px in leftover[r]:
        if (px[3] != 0):
            trans = False
            break
    if (trans):
        leftover = np.delete(leftover, r, 0)
        l = len(leftover)
    else:
        r += 1
c = 0
w = len(leftover[0])
while (c < w):
    trans = True
    for row in range(0, len(leftover)):
        if (leftover[row][c][3] != 0):
            trans = False
            break
    if (trans):
        leftover = np.delete(leftover, c, 1)
        w = len(leftover[0])
    else:
        c += 1
image_show(leftover)
    
    #io.imsave("frag\\frag" + str(count) + ".png", im3d)
    #image_show(color.label2rgb(image_slic, image, kind='avg'))