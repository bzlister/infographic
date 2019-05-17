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


def image_show(split_image):
    plt.imshow(split_image)
    plt.show()

image = io.imread('images\\ig1.png')
transparent_px = [0,0,0]
image_slic = seg.slic(image,n_segments=6) #need to tune this parameter for every image
borders = seg.find_boundaries(image_slic)

#Partition image into sub-images based on border
i = 0
sub_images = []
sub_image = []
offset = 0
np.append(borders, len(borders[-1])*[True])
while (i < len(borders)):
    sub_image.append([])
    j = 0
    while (j < len(borders[i])):
        if (borders[i][j] == False):
            if (j > len(sub_image[i-offset])):
                sub_image[i-offset].append([transparent_px]*(j-len(sub_image[i-offset]))) #left-pad with null values
            sub_image[i-offset].append(image[i][j].tolist())
            j += 1
        else:
            break
    if (j == 0):
        sub_images.append(sub_image[:-1])
        sub_image = []
        offset = i+1
    i += 1
sub_images.append(sub_image)

#Right-pad with null values
for img in sub_images:
    cluster = []
    longest = 0
    for row in img:
        if (len(row) > longest):
            longest = len(row)
    for r in img:
        if (r != []):
            try:
                cluster.append(r + [transparent_px]*(longest-len(r)))
            except:
                cluster.append([r] + [transparent_px]*(longest-len(r)))
    if (longest != 0):
        im3d = np.empty([len(cluster), len(cluster[0]), 3])   
        for u in range(0, len(cluster)):
            for v in range(0, len(cluster[u])):
                try:
                    im3d[u][v] = cluster[u][v]
                except:
                    print(cluster[u])
                    x = 6/0
        image_show(im3d)
    #io.imsave("frag\\frag" + str(count) + ".png", im3d)
    #count += 1
    #image_show(color.label2rgb(image_slic, image, kind='avg'))