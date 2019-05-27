import os
import image_segment
import extract_text
from skimage import io
#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, Flatten
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import rescale
import re

out = 'C:\\Users\\bzlis\\Documents\\infographic\\train'
src = 'C:\\Users\\bzlis\\Documents\\infographic\\images'
transparent_px = np.uint8([0,0,0,0])

def dim():    
    heights = []
    widths = []
    for f in os.listdir(out):
        for data in os.listdir(out + '\\' + f):
            if ('c' in data):
                img = io.imread(out + '\\' + f + '\\' + data)
                heights.append(len(img))
                widths.append(len(img[0]))
    plt.hist(heights)
    plt.title("Heights")
    plt.show()
    plt.hist(widths)
    plt.title("Widths")
    plt.show()
    print('Average height: %f' %(sum(heights)/len(heights))) #637
    print('Average width: %f' %(sum(widths)/len(widths))) #813

def reshape(h=127, w=163):
    for f in os.listdir(out):
        print(f)
        for data in os.listdir(out + '\\' + f):
            if ('c' in data):
                img = io.imread(out + '\\' + f + '\\' + data)
                if ((len(img) != h) | (len(img[0]) != w)):
                    if (h/len(img) >= w/len(img[0])):
                        scaling_factor = w/len(img[0])
                        img = rescale(img, scaling_factor, anti_aliasing=True, multichannel=True)
                        if (h - len(img) > 0):
                            img = np.vstack((img, np.full((h-len(img), len(img[0]), 4), transparent_px)))
                    else:
                        scaling_factor = h/len(img)
                        img = rescale(img, scaling_factor, anti_aliasing=True, multichannel=True)
                        if (w - len(img[0]) > 0):
                            img = np.hstack((img, np.full((len(img), w-len(img[0]), 4), transparent_px)))
                if ((len(img) != h) | (len(img[0]) != w)):
                    print("Error with %s! Height: %d Width: %d" %(f + '\\' + data, len(img), len(img[0])))
                else:
                    io.imsave(out + '\\' + f + '\\' + data, img)
                



def grow():
    #[maxw, maxh] = dim()
    maxW = 4651
    maxH = 4651
    for f in os.listdir(out):
        print(f)
        for data in os.listdir(out + '\\' + f):
            if ('c' in data):
                img = io.imread(out + '\\' + f + '\\' + data)
                delH = maxH - len(img)
                delW = maxW - len(img[0])
                img = np.vstack((img, np.full((delH, len(img[0]), 4), transparent_px)))
                img = np.hstack((img, np.full((len(img), delW, 4), transparent_px)))
                io.imsave(out + '\\' + f + '\\' + data, img)
        
def shrink():
    for f in os.listdir(out):
        print(f)
        for data in os.listdir(out + '\\' + f):
            if ('c' in data):
                img = Image.open(out + '\\' + f + '\\' + data)
                img.crop(img.getbbox()).save(out + '\\' + f + '\\2' + data)



def rename():
    for f in os.listdir(out):
        print(f)
        for data in os.listdir(out + '\\' + f):
            if ('2c' in data):
                os.remove(out + '\\' + f + '\\' + data[1:])
                os.rename(out + '\\' + f + '\\' + data, out + '\\' + f + '\\' + data[1:])

def generate_train_data():
    count = 0
    for f in os.listdir(src):
        num = 0
        bg = 0
        try:
            os.mkdir(out + '\\img' + str(count))
            fragments = image_segment.fragment(src + '\\' + f)
            for chunk in fragments:
                text = extract_text.pull(chunk)
                uniform_chunk = image_segment.normalize(chunk)
                if (text != ''):
                    outfile = open(out + '\\img' + str(count) + '\\t' + str(num) + '.txt', 'w')
                    outfile.write(text)
                    outfile.close()
                    io.imsave(out + '\\img' + str(count) + '\\c' + str(num) + '.png', uniform_chunk)
                    num += 1
                else:
                    io.imsave(out + '\\img' + str(count) + '\\bg' + str(bg) + '.png', uniform_chunk)
                    bg += 1
        except:
            num = 0
        count += 1
        print(count)

def load(dir, h=127, w=163):
    text = []
    comp = []
    #bg = []
    for data in os.listdir(dir):
        #if ('t' in data):
         #   txt_file = open(dir + '\\' + data, 'r')
          #  text.append(txt_file.read().split())
           # txt_file.close()
        if ('c' in data):
            img = io.imread(dir + '\\' + data)
            comp.append(encode(img))
    #return [text, comp]
    return comp
            
def encode(img, h=127, w=163):
    new_img = np.full((h,w), 0)
    for r in range(0, len(img)):
        for c in range(0, len(img[0])):
            new_img[r][c] = image_segment.hash(img[r][c])
    return new_img

def decode(img, h=127, w=163):
    new_img = np.full((h,w, 4), transparent_px)
    for r in range(0, len(img)):
        for c in range(0, len(img[0])):
            new_img[r][c] = image_segment.unhash(img[r][c])
    return new_img

def requestRecords():
    train_comp = []
    test_comp = []
    train_txt = []
    test_txt = []
    for fname in os.listdir(out):
        #if ((i >= start*batch_size) & (i < (start+1)*batch_size)):
        record = load(out + '\\' + fname)
        #train_comp += record[1]
        train_comp += record
        #i += 1
    train_comp = np.array(train_comp)
    train_comp = train_comp.reshape(len(train_comp), 127*163)
    #train_comp = (train_comp.astype(np.float32) - 127.5)/127.5
    #print(np.shape(train_comp))
    return train_comp
            
""" 

def build_GAN():
    #create model
    model = Sequential()
    
    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    [train_txt, train_comp, test_txt, test_comp] = partition()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3) """