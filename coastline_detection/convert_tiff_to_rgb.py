from skimage import img_as_ubyte, io, img_as_uint
import numpy as np
from skimage.filters import threshold_isodata


def convert_tiff_to_rgb_int8(path_to_image, new_name="new image.png", save_img=False):
    tiff_image = io.imread(path_to_image)
    temp = np.stack(([tiff_image[0], tiff_image[1], tiff_image[2]]), axis=2)
    temp = img_as_ubyte(temp)
    if save_img:
        io.imsave(new_name, temp)
    else:
        return temp


def convert_tiff_to_rgb_int16(path_to_image, new_name="new image.png", save_img=False):
    tiff_image = io.imread(path_to_image)
    temp = np.stack(([tiff_image[0], tiff_image[1], tiff_image[2]]), axis=2)
    if save_img:
        io.imsave(new_name, temp)
    else:
        return temp


def ndwi_grayscale_float_and_uint8(path_to_iamge, name="ndwi_image.png"):
    image = io.imread(path_to_iamge)
    ndwi_im = (image[1] - image[4]) / (image[1] + image[4])
    ndwi_uint8 = img_as_ubyte(ndwi_im)
    io.imsave("float_" + name, ndwi_im)
    io.imsave("uint8_" + name, ndwi_uint8)


def ndwi_grayscale_float_and_uint8_with_sample_selectors(path_to_iamge, name="ndwi_image.png"):
    image = io.imread(path_to_iamge)
    ndwi_im = (image[1] - image[4]) / (image[1] + image[4])
    ndwi_im = ndwi_im >= 0.3
    ndwi_uint8 = img_as_ubyte(ndwi_im)
    io.imsave("float_selector" + name, ndwi_im)
    io.imsave("uint8_selector" + name, ndwi_uint8)


def ndwi_1_grayscale_float_and_uint8(path_to_iamge, name="ndwi_image.png"):
    image = io.imread(path_to_iamge)
    ndwi_im = (image[4] - image[3]) / (image[4] + image[3])
    ndwi_uint8 = img_as_ubyte(ndwi_im)
    io.imsave("float_op1_" + name, ndwi_im)
    io.imsave("uint8_op1_" + name, ndwi_uint8)


def ndwi_1_grayscale_float_and_uint8_with_sample_selectors(path_to_iamge, name="ndwi_image.png"):
    image = io.imread(path_to_iamge)
    ndwi_im = (image[4] - image[3]) / (image[4] + image[3])
    ndwi_im = ndwi_im >= 1
    ndwi_uint8 = img_as_ubyte(ndwi_im)
    io.imsave("float_op1_selector" + name, ndwi_im)
    io.imsave("uint8_op1_selector" + name, ndwi_uint8)


def iosdata_thres_on_ir_layer(path_to_image, name="isodata.png"):
    """
    Does not work very well on low landmass images. So the island image that I have is just black.
    I think that there is just not enough data for the isodata to work well

    :param path_to_image:
    :param name:
    :return:
    """
    tif = io.imread(path_to_image)
    ir_layer = tif[4]
    thres = threshold_isodata(ir_layer)
    result = ir_layer > thres
    io.imsave("isodata/" + name, img_as_uint(result))


iosdata_thres_on_ir_layer("Oliktok OVR Files/671610_2013-07-21_RE5_3A_Analytic_clip.tif")

iosdata_thres_on_ir_layer('Oliktok OVR Files/671710_2013-07-21_RE5_3A_Analytic_clip.tif', name="isodata2.png")
"""
ndwi_grayscale_float_and_uint8('/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/'
                               'Oliktok OVR Files/671610_2013-07-21_RE5_3A_Analytic_clip.tif')

ndwi_grayscale_float_and_uint8_with_sample_selectors('/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/'
                                                     'Oliktok OVR Files/671610_2013-07-21_RE5_3A_Analytic_clip.tif')


ndwi_1_grayscale_float_and_uint8('/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/'
                                 'Oliktok OVR Files/671610_2013-07-21_RE5_3A_Analytic_clip.tif')


ndwi_1_grayscale_float_and_uint8_with_sample_selectors('/home/nelson/PycharmProjects'
                                                       '/coastline_detection_with_LANDSAT/Oliktok OVR Files/'
                                                       '671610_2013-07-21_RE5_3A_Analytic_clip.tif')

##############################################################################################################

ndwi_grayscale_float_and_uint8('/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/'
                               'Oliktok OVR Files/671710_2013-07-21_RE5_3A_Analytic_clip.tif', name="ndwi_image2.png")

ndwi_grayscale_float_and_uint8_with_sample_selectors('/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/'
                                                     'Oliktok OVR Files/671710_2013-07-21_RE5_3A_Analytic_clip.tif', name="ndwi_image2.png")


ndwi_1_grayscale_float_and_uint8('/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/Oliktok OVR Files/'
                                 '671710_2013-07-21_RE5_3A_Analytic_clip.tif', name="ndwi_image2.png")


ndwi_1_grayscale_float_and_uint8_with_sample_selectors('/home/nelson/PycharmProjects/'
                                                       'coastline_detection_with_LANDSAT/'
                                                       'Oliktok OVR Files/'
                                                       '671710_2013-07-21_RE5_3A_Analytic_clip.tif', name="ndwi_image2.png")
"""


# output /home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/Oliktok OVR Files/671610_2013-07-21_RE5_3A_Analytic_clip.tif

# output2  /home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/Oliktok OVR Files/671710_2013-07-21_RE5_3A_Analytic_clip.tif


