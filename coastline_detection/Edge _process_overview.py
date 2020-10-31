# !/usr/bin/env python
# coding: utf-8

# # Edge Detection on Plant Imagery for North Alaskan Coast
# This project is designed to create a method of detecting coastline
# in a set of given satellite imagery from planet labs. Specifically
# for the following code the coastlines that are being detected are 
# located in northern Alaska. With the goal for this project being
# the accurate coastline detection of over 1000 km of northern Alaskan
# coastline to measure coastal erosion.
# 
# To do this the following edge detection and threshold methods have
# been used: Canny edge detection, Sobel Edge Detection, 
# Normalized Difference Water Index (NDWI), Holistically-Nested
# Edge Detection, and ISODATA clustering. Each of these methods have
# their strong and weak points when it comes to clustering. I will go
# over a outline of how each method work, show the method results on 
# two sample images from the north Alaskan coast, and describe why
# the method was either chosen for further use in the project or 
# dropped.

# # Canny Edge Detection
# Canny edge detection is a multi-stage process used to detect edge on
# a wide array of images.
# 
# Step 1:
# We first have to import opencv and numpy in addition to handle the images
# then load in the image as grayscale 

# In[ ]:


import cv2
import numpy as np
# image = "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg"
image = "good_images/output2.jpg"
image = cv2.imread(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # cv2.COLOR_BGR2GRAY one of opencv many color change options


# ![](Edge_process_overview images/Original Grayscale.png "Original Grayscale")
# Now we need to apply a blur to the image. I know that applying a blur
# seems counter intuitive but it is useful for removing noise at the 
# edge locations of the image. For this example we will be using a 
# gaussian blur of size 5x5. This should remove most of the noise in
# our image.

# In[ ]:


blur = cv2.GaussianBlur(image, (5, 5), 0)


# ![](Edge_process_overview images/Blurred.png "Blurred")
# Now we can apply canny edge detection on the blurred image. Part of 
# canny edge detection is entering in the upper and lower bounds of 
# the pixels in the image. Any pixels with a value higher than the upper
# bound is automatically considered an edge. While any pixels below the
# lower bound threshold are considered to be non-edges and are to be 
# discarded. Values between the two thresholds are considered to be either
# edges or none edges based on their connectivity to other pixels. 
# One method about thinking of this process is that edges are the 
# points where two gaussian/bell curves meet.

# In[ ]:


canny = cv2.Canny(blur, 30, 150)  # these values are just random


# ![](Edge_process_overview images/Canny.png "Canny")
# As you can see in this image canny edge detection works pretty well.
# You can see continuous edges from the coastline. It also picks up 
# the ponds and lakes that are also in the image. The main problem with
# Canny Edge detection though is that it relies way to heavily on human
# inputted values for determining edges. This makes trying to use this process
# a poor choice for large amounts of images when a large variety of 
# different conditions exist for those images. Exactly the situation for
# this project. An example of this is the following image with Canny Edge
# detection applied. 
# ![](Edge_process_overview images/Original2 Grayscale.png "image 2")
# Canny edge applied with the same values
# ![](Edge_process_overview images/canny2.png "Canny on image 2")
# You'll see in this image that in the upper left corner parts of the 
# island end up getting cut off. 
# 
# In the end though Canny edge is still a really good edge detection choice
# in most cases

# # Sobel Edge Detection
# Sobel Edge Detection is an approximation of the derivative of the image separated in
# the x and y directions.  These gradients are then added together. 
# 
# Step 1:
# This is also an opencv function. Lucky we already have opencv imported.
# We are also going to be using a grayscale image again. Which is still imported
# as the image object. We then apply the opencv sobel function to get the x and y
# gradient. To perform this operation we also need to convert the int8 pixels into 64 bit floats.
# Otherwise information will be lost since we are calculating gradients.

# In[ ]:


sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)


# ![](Edge_process_overview images/sobelx.png "Sobel X gradient")
# ![](Edge_process_overview images/sobely.png "Sobel Y gradient")
# Step 2:
# we then have to convert the 64 float values back into int8. So that we have pixel
# values that are readable for image displaying. Followed by adding the x and y gradients together.

# In[ ]:


sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)


# ![](Edge_process_overview images/sobelCombined.png "Sobel Combined gradient")
# AS you can see this image shows the edges really well. The main problem though is that
# there is a bunch of additional noise that is also in the image. This noise shows up in
# the area of the image where ocean would normally be, Getting rid of these noisy pixels is
# necessary when an edge detection process is required to to have a high level accuracy.
# Thus sort of ruling it out for accurate coastline measurements. The reason that I say sort of is that
# Sobel edge detection is part of Canny Edge detection. The part of Canny Edge that 
# operates between the threshold parameters contains Sobel edge detection within it.
# Which is why Sobel can be useful. It gives you the possibility to detect edges that
# you otherwise might miss with Canny Edge.
# 
# # ISODATA Thresholding
# ISODATA thresholding is a clustering algorithm that takes the histogram of the entire image
# and divides them into two separate groups of data. In our case it separates the water from the land.
# It matters in this case as to what band layer that this process is applied to. The near infrared 
# is the layer that gets the best results due to the greatest difference between water and land.
# 
# Step 1:
# For this process we need to use the sckit-image package which contains the appropriate 
# functions to read a tiff file and an implemented ISODATA function.

# In[1]:


from skimage import io, img_as_uint
from skimage.filters import threshold_isodata
tiff_image = "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"              "Oliktok OVR Files/671610_2013-07-21_RE5_3A_Analytic_clip.tif"
tif = io.imread(tiff_image)  # importing the tiff file
ir_layer = tif[4]  # the NIR layer of the tiff layer


# Step 2:
# We then need to get the ISODATA threshold value. Followed with applying it to the 
# original image. The resulting image is then saved in a png file which requires the
# conversion of the int16 pixel values to the int8 pixel values

# In[ ]:


thres = threshold_isodata(ir_layer)
result = ir_layer > thres
io.imsave("isodata_thres.png", img_as_uint(result))


# ![](Edge_process_overview images/isodata_thres.png "ISODATA result")
# As you can see in the resulting image ISODATA has one of the best performing
# coastline detection methods. Pulling out a continuous contour of the 
# coastline. Which an edge can easily be extracted through the used of Canny Edge.
# This is only the case when there are enough land pixels in an image
# though. If there are not enough land pixels in an image you instead get a mess 
# of static noise in the image. This is easily with a change in the image file.

# In[ ]:


tiff_image_bad = "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"                  "Oliktok OVR Files/671710_2013-07-21_RE5_3A_Analytic_clip.tif"
tif = io.imread(tiff_image_bad)  # importing the tiff file
ir_layer = tif[4]  # the NIR layer of the tiff layer
thres = threshold_isodata(ir_layer)
result = ir_layer > thres
io.imsave("isodata_thres_bad.png", img_as_uint(result))


# ![](Edge_process_overview images/isodata_thres_bad.png "ISODATA result bad")
# As you can see this image does not really show the coastline in this image.
# 
# In the end though out of all the methods that have been applied in this process
# ISODATA has the best performance. Provided that enough land pixels have been
# included in the original image. It just means that when applying ISODATA threshold
# to an image make sure to test the result with another process to make sure no
# mistakes were made.
# 
# # Normalized Difference Water Index (NDWI)
# NDWI uses the properties of the different band layers in the tiff files to 
# determine the difference between water and land. It does this by taking 
# advantage of simple matrix math. 
# 
# Step 1:
# Use the image that was previously imported for the ISODATA threholding and 
# perform the following math.
# NDWI = (green band - near infrared band) / (green band + near infrared band)

# In[ ]:


ndwi_im = (tif[1] - tif[4]) / (tif[1] + tif[4])


# ![](Edge_process_overview images/ndwi.png "NDWI results")
# As you can see in the resulting image when it comes to coastline extraction 
# NDWI does not really perform all that well. It really breaks down when 
# viewing the shore between the water and the land. This is due to NDWI's strong
# point not being accuracy in determining the boundaries between water and land but
# instead determining if there is water in an image. Making it a 
# process that is likely not to be used very often in this project.
# 
# # Holistically-Nested Edge Detection
# Holistic Edge is the final edge detection process that was implemented for this
# project. It is the only edge detection process that utilizes a pre-trained neural network.
# By using that neural network Holistic Edge is able to learn the hierarchical representations
# that represent the edges in the image. Allowing for higher accuracy in performance on 
# general edge detection problems than any other method of edge detection in this project.
# 
# Step 1:
# We first have to import in the pre-trained model that will be used.
# This is done in this project by using caffe and opencv.

# In[ ]:


image = "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg"
image = cv2.imread(image)
holistic_proto_path = "hed_model/deploy.prototxt"
holistic_model_path = "hed_model/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(holistic_proto_path, holistic_model_path)  # read model in


# Step 2:
# We now need to create a class that we can use to crop layers of the network.
# Allowing us to derive in input shape, batch size, input channels, height and target width.
# Which are then passed to each layer in the neural network.

# In[ ]:


class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


# Step 3:
# Now we construct a blob from out image that we pass through the holistic neural network.
# Which after rescaling the resulting pixels back to int8 we have our edge image.

# In[ ]:


(H, W) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")


# ![](Edge_process_overview images/holistic.png "Holistic results")
# Holistic works really well at actually finding the edges of the images. It even works
# in cases with the islands that were shown in the second ISODATA image. On the other 
# hand it does not really do accuracy really well. As you can see the resulting edges on the image
# are several pixels wide. Which prevents holistic from being the method of choice for coastline
# extraction. It does make Holistic the best tool to test any of the other methods to ensure that 
# they are not failing in there process.
# 
# # Planet Imagery Conversions
# This is not an edge detection process. It instead is about converting the TIFF files that 
# represent the data from Planet labs to a format that is usable by all of the opencv 
# methods that are used in this project. I'm including this because this is currently done
# using terminal commands that are given to gdal_translate. I attempted to convert this process
# to python but was unable to do so. If you are able to figure this out I HIGHLY RECOMMEND THAT 
# YOU CHANGE THE CURRENT CODE.
# 
# Anyway here are the commands and what they are doing. Which I will not actually show in the
# python console since the only alteration is wrapping the command in a sys.system call.
# 
# First command:
# gdal_translate tiff_file intermediate_tif -b 3 -b 2 -b 1 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB
# This command takes out the first three bands of the original tiff file and moves them to correct band
# structure for an RBG image
# 
# gdalinfo -mm intermediate_tif
# The next command is used to obtain the pixel min and max for each of the bands. The min and max for the
# entire file are then used in the below command.
# 
# gdal_translate intermediate_tif scaled_tif -scale 570 23800 0 65535 -exponent 0.5 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB
# Now the tif file has its pixel values rescaled to values of a correct scale for showing color images.
# If this command is not run the image appears very dark.
# 
# gdal_translate -of JPEG -scale -co worldfile=yes scaled_tif final_file_name
# Finally this last command translates the tif file into a JPEG file.
# 
# 
# # Help Installing opencv
# Installing opencv can be a major pain. There are some binary packages that you can install
# but one of the problems with those is that sometimes are unable to perform I/O or other
# important functions. So I'm including some resources to help you install opencv
# 
# The easiest way that I found is to install __[Anaconda](https://www.anaconda.com/)__
# then use the following command in the terminal
# conda install -c conda-forge opencv
# 
# Otherwise the following are some good tutorials to install opencv 
# __[pyimagesearch](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)__
# __[pydeeplearning](https://pydeeplearning.com/opencv/install-opencv-with-c-on-ubuntu-18-04/)__
# 
# Take the easy way is what I recommend since install opencv is highly machine 
# dependent so it is very easy to run into problems.
# 
# 
# # Wrap up
# I hope that this was enough for you to be able to have an overview of all of the edge 
# detection methods that I have implemented. I know that the next stage of the project is
# to output the edge results of some of these processes to the band layers of the 
# original tiff files. I recommend using ISODATA with Holistic as a testing backup.
# While possible including Canny Edge if you want to. Just find a way to set the 
# threshold values that works.
# 
# The resulting band layers should be enough to allow you to construct vectors that you
# will need to began to actually track the coastlines over a time series, Good luck.
# 
