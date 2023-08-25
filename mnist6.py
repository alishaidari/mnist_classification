# Read MNIST training data set and store it as a 60000 by 784 matrix.
# There are 60000 training images, each of which is 28 x 28 = 784 pixels
# Read MNIST test data set and store it as a 10000 by 784 matrix.
# There are 10000 test images, each of which is 28 x 28 = 784 pixels
# plot a test image as well as the 3 nearest training images

import sys
import numpy as np
import matplotlib.pyplot as plt

if (len(sys.argv) != 7):
    print ("Not enough command line arguments")
    quit()

test_idx = int(sys.argv[1])
train_idx1 = int(sys.argv[2])
train_idx2 = int(sys.argv[3])
train_idx3 = int(sys.argv[4])
train_idx4 = int(sys.argv[5])
train_idx5 = int(sys.argv[6])


f = open('t10k-images-idx3-ubyte','rb')

image_size = 28
num_test_images = 10000
f.read(16)
buf = f.read(image_size * image_size * num_test_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test = data.reshape(num_test_images, image_size * image_size)

f = open('train-images-idx3-ubyte','rb')

image_size = 28
num_train_images = 60000
f.read(16)
buf = f.read(image_size * image_size * num_train_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
train = data.reshape(num_train_images, image_size * image_size)

plt.rcParams['figure.figsize'] = (20, 20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
image1 = np.asarray(test[test_idx]).reshape(image_size,image_size)
image2 = np.asarray(train[train_idx1]).reshape(image_size,image_size)
image3 = np.asarray(train[train_idx2]).reshape(image_size,image_size)
image4 = np.asarray(train[train_idx3]).reshape(image_size,image_size)
image5 = np.asarray(train[train_idx4]).reshape(image_size,image_size)
image6 = np.asarray(train[train_idx5]).reshape(image_size,image_size)

f, axarr = plt.subplots(1,6)
axarr[0].imshow(image1,cmap='gray',vmin=0, vmax=255, interpolation='none')
axarr[1].imshow(image2,cmap='gray',vmin=0, vmax=255, interpolation='none')
axarr[2].imshow(image3,cmap='gray',vmin=0, vmax=255, interpolation='none')
axarr[3].imshow(image4,cmap='gray',vmin=0, vmax=255, interpolation='none')
axarr[4].imshow(image5,cmap='gray',vmin=0, vmax=255, interpolation='none')
axarr[5].imshow(image6,cmap='gray',vmin=0, vmax=255, interpolation='none')
plt.savefig("mnist6.png",bbox_inches='tight')
