# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from skimage import feature
from skimage import exposure
import cv2
import json
import os


# %%
with open('train.json') as f:
    data = json.load(f)


# %%
price = data['price']
latitude = data['latitude']
longitude = data['longitude']
time = data['created']
pic = data['photos']


# %%
def u_to_image(url):
	response = requests.get(url)
	img = Image.open(BytesIO(response.content))
	# return the image
	return img


# %%
def image_to_hog(img):
    h = feature.hog(img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),transform_sqrt=True,block_norm="L1")
    return h


# %%
len(pic)


# %%
pic['4']


# %%
img = u_to_image(https://photos.renthop.com/2/7170325_3bb5ac84a5a10227b17b273e79bd77b4.jpg)
h,hi = feature.hog(img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),transform_sqrt=True,visualize = True,block_norm="L1")
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image", hi)


# %%
def resize(img):
    width, height = img.size  
    width, height = img.size  
    left = 4
    top = height / 5
    right = 154
    bottom = 3 * height / 5
    newsize = (round(width/4), round(height/4))
    im1 = img.crop((left, top, right, bottom))
    im1 = im1.resize(newsize)
    return im1


# %%
def append_record(record):
    f = open("my_file.txt", "a+")
    f.write(str(record))
    f.close()


# %%
# for x in pic:
#     image_feature = {}
#     tmp = []
#     for y in pic[x]:
#         try:
#             img = u_to_image(y)
#             tmp.append(image_to_hog(resize(img)))
#             del img
#         except (OSError,NameError):
#             print(y)
#     image_feature[int(x)]=tmp
#     append_record(image_feature)
# #     del tmp


# %%
for x in pic:
    image_feature = {}
    tmp = []
    for y in pic[x]:
        try:
            img = u_to_image(y)
            tmp.append(image_to_hog(resize(img)))
            del img
        except (OSError,NameError):
            print(y)
    image_feature[int(x)]=tmp
    y = np.load("save.npy",allow_pickle=True) if os.path.isfile("save.npy") else [] #get data if exist
    np.save("save.npy",np.append(y,image_feature))
    del tmp


# %%
# xl = np.load("save.npy",allow_pickle=True)
# list(xl)


# %%


