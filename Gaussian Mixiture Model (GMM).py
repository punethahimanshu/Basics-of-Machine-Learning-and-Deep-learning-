
"""
@author: HIMANSHU
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2

img = cv2.imread("Images/BSE.tif")
plt.imshow(img)

# convert M x N x 3 image into K x 3 where K = M x N

img2 = img.reshape(-1,3)       # -1 means M x N

img2 = np.float32(img2)

from sklearn.mixture import GaussianMixture as GMM
# Covarience choices- full, tied, diag, spherical

gmm_model = GMM(n_components = 4, covariance_type = 'tied').fit(img2)
gmm_label = gmm_model.predict(img2)

# Reshape the image to construct segmented image

original_shape = img.shape
segmented_image = gmm_label.reshape(original_shape[0], original_shape[1])
plt.imshow(segmented_image)
cv2.imwrite("Images/segmented.jpg", segmented_image )

########################################################################

#     Finding best number of components/cluster/segment

#     Using Bayesian information criterion (BIC) to find the best
#     number of components

bic_value = gmm_model.bic(img2)
print(bic_value)

n_components = np.arange(1,10)
gmm_models = [GMM(n, covariance_type='tied').fit(img2) for n in n_components]
plt.plot(n_components, [m.bic(img2) for m in gmm_models], label='BIC')
