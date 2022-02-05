# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:06:37 2022

@author: HIMANSHU
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from skimage import io
import numpy as np

img = io.imread("Images/BSE.tif", as_gray= False)
plt.imshow(img, cmap='gray')

# convert M x N x 3 image into Kx3, where K = M x N (653x734)

img2 = img.reshape(-1,3)      #-1 means MxN = 479302

# we need to convert uint8 values into float values as it is the requirement 
# for K-Means Method

img2 = np.float32(img2)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init =10, max_iter = 300,
                random_state = 42)
model = kmeans.fit(img2)
predicted_values = kmeans.predict(img2)

segm_image = predicted_values.reshape((img.shape[0], img.shape[1]))
plt.imshow(segm_image, cmap ='gray')
