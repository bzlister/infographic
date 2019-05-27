import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import skimage.draw as draw
import skimage.color as color
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import label, regionprops
import math
import cv2
import os
from skimage.filters import median


def image_show(split_image):
    plt.imshow(split_image)
    plt.show()

def normalize(component):
    cmap = {}
    wdel = 1
    hdel = 1
    r = 0
    while (r < len(component)):
        c = 0
        while (c < len(component[r])):
            px = component[r][c]
            if (px[3] != 0):
                h = hash(component[r][c])
                try:
                    cmap[h] += 1
                except:
                    cmap[h] = 1
            c += wdel
        r += hdel
    count = 0
    mode = 0
    for color in cmap:
        if (cmap[color] > count):
            mode = color
            count = cmap[color]
    dominant_color = unhash(mode)
    for r in range(0, len(component)):
        for c in range(0, len(component[r])):
            if (component[r][c][3] != 0):
                component[r][c] = dominant_color
    return component

def hash(color):
    return 256*(256*color[0] + color[1]) + color[2]

def unhash(h):
    b = h%256
    g = (h//256)%256
    r = (h//256//256)%256
    return np.uint8([r,g,b,255])

def fragment(path):
    image = io.imread(path)
    transparent_px = np.uint8([0,0,0,0])
    image_slic = []
    try:
        image_slic = seg.slic(image, n_segments=15, convert2lab=True)
    except:
        image_slic = seg.slic(color.rgba2rgb(image), n_segments=15, convert2lab=True)
    regions = regionprops(image_slic)
    leftover = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2RGBA)
    count = 0
    fragments = []
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
        #io.imsave("frag\\frag" + str(img_id) + "\\clip" + str(count) + ".png", clip)
        fragments.append(clip)
        count += 1

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
    fragments.append(leftover)
    return fragments