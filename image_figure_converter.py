import numpy as np
import matplotlib.pyplot as plt

# code for displaying multiple images in one figure

#import libraries
import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(10, 10))

# setting values to rows and column variables
rows = 2
columns = 3
img_1_path = '/home/berk/Desktop/Tez/compression-main/experiments/test_photo/org/kodim19.png'
img_2_path = '/home/berk/Desktop/Tez/compression-main/experiments/test_photo/117_27.7_0.142.png'
img_3_path = '/home/berk/Desktop/Tez/compression-main/experiments/test_photo/435_30.86_0.35.png'
img_4_path = '/home/berk/Desktop/Tez/compression-main/experiments/test_photo/835_32_54_0.52.png'
img_5_path = '/home/berk/Desktop/Tez/compression-main/experiments/test_photo/3140_36.48_1.079.png'
img_6_path = '/home/berk/Desktop/Tez/compression-main/experiments/test_photo/11700_40.56_1.9.png'



# reading images
Image1 = cv2.cvtColor(cv2.imread(img_1_path), cv2.COLOR_BGR2RGB)
Image2 = cv2.cvtColor(cv2.imread(img_2_path), cv2.COLOR_BGR2RGB)
Image3 = cv2.cvtColor(cv2.imread(img_3_path), cv2.COLOR_BGR2RGB)
Image4 = cv2.cvtColor(cv2.imread(img_4_path), cv2.COLOR_BGR2RGB)
Image5 = cv2.cvtColor(cv2.imread(img_5_path), cv2.COLOR_BGR2RGB)
Image6 = cv2.cvtColor(cv2.imread(img_6_path), cv2.COLOR_BGR2RGB)

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("Original Image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(Image2)
plt.axis('off')
plt.title("27.7dB, 0.14bpp")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(Image3)
plt.axis('off')
plt.title("30.86dB, 0.35bpp")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(Image4)
plt.axis('off')
plt.title("32.54dB, 0.52bpp")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 5)

plt.imshow(Image5)
plt.axis('off')
plt.title("36.48dB, 1.08bpp")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 6)

plt.imshow(Image6)
plt.axis('off')
plt.title("40.56dB, 1.90bpp")

plt.show()
