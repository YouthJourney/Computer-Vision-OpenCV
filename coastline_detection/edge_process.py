from skimage import io, img_as_uint
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_isodata
import os
import gdal
import time


class makeEdge:
    """
    class is designed to take in a tiff file. Convert it to a JPEG file then import it numpy arrays using OpenCv.
    At that point 
    """
    def __init__(self, tiff_file):
        self.image = self.convert_tiff_to_png(tiff_file)
        self.holistic_proto_path = "hed_model/deploy.prototxt"
        self.holistic_model_path = "hed_model/hed_pretrained_bsds.caffemodel"

    def blend_images(self, im_2, alpha=0.7, name="blended.png"):
        src1 = self.image
        src2 = cv2.imread(im_2)
        if src1.shape != src2.shape:
            raise ValueError("Image shape does not batch")

        # for lack of a better way to solve this
        for x in range(src2.shape[0]):
            for y in range(src2.shape[1]):
                src2[x, y] = [0, src2[x, y, 0], src2[x, y, 0]]

        beta = (1.0 - alpha)
        dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
        cv2.imwrite("edge_results/blend/" + name, dst)

    def grayscale_with_color_bar(self, colormap="gist_gray", name="grayscale with colorbar.png"):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        plt.imshow(image, cmap=colormap)
        plt.colorbar()
        plt.savefig("grayscale_colorbar/" + name)
        plt.show()

    def run_a_bunch_of_canny_edge_samples(self, blur, upper_bound_list, lower_bound_list, name=""):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        im = cv2.GaussianBlur(image, blur, 0)

        row_count = 0
        col_count = 0
        for col in upper_bound_list:
            for row in lower_bound_list[col_count]:
                canny = cv2.Canny(im, row, col)
                print("Col " + str(col_count) + "    row " + str(row_count))
                cv2.imwrite(name + "/Upper " + str(col) + " , lower " + str(row) + ".png", canny)
                row_count += 1
            row_count = 0
            col_count += 1

    def test_for_canny_edge_samples(self, name=""):
        u = []
        l = []

        for i in range(20, 210, 10):
            u.append(i)
            temp = []
            for j in range(0, i, 10):
                temp.append(j)
            l.append(temp)
        self.run_a_bunch_of_canny_edge_samples((5, 5), u, l, name=name)

    def sobel(self, name="sobel.png"):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobelCombined = cv2.bitwise_or(sobelX, sobelY)

        cv2.imwrite(name, sobelCombined)

    def canny_edge(self, blur, canny_lower, canny_upper, name="canny_edge.png"):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        image = cv2.GaussianBlur(image, blur, 0)
        canny = cv2.Canny(image, canny_lower, canny_upper)
        cv2.imwrite(name, canny)

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

    def holistic_edge(self, name="holistic.png"):
        net = cv2.dnn.readNetFromCaffe(self.holistic_proto_path, self.holistic_model_path)
        cv2.dnn_registerLayer("Crop", self.CropLayer)
        (H, W) = self.image.shape[:2]
        blob = cv2.dnn.blobFromImage(self.image, scalefactor=1.0, size=(W, H),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")
        cv2.imwrite(name, hed)

    def ndwi(self, tiff_image, name="ndwi.png"):
        image = io.imread(tiff_image)
        ndwi_im = (image[1] - image[4]) / (image[1] + image[4])
        io.imsave("float_" + name, ndwi_im)

    def iosdata_thres_on_ir_layer(self, tiff_image, name="isodata.png"):
        """
        Does not work very well on low landmass images. So the island image that I have is just black.
        I think that there is just not enough data for the isodata to work well

        :param path_to_image:
        :param name:
        :return:
        """
        tif = io.imread(tiff_image)
        ir_layer = tif[4]
        thres = threshold_isodata(ir_layer)
        result = ir_layer > thres
        io.imsave("isodata/" + name, img_as_uint(result))

    def convert_tiff_to_png(self, tiff_file, intermediate_tif="output.tif", scaled_tif="output_scaled.tif",
                            final_file_name="out.jpg"):
        """
        This function right here is one that I HIGHLY RECOMMEND THAT IT BE WRITTEN TO PURE PYTHON.
        I have not done this at this time due to the time it would take me due to my lack of knowledge of gdal
        along with my lack of knowledge of just gis in general. Also just because the code below works.
        STILL IF YOU CAN REWRITE THE DAMN THING

        reference source: https://medium.com/planet-stories/a-gentle-introduction-to-gdal-part-4-working-with-satellite-data-d3835b5e2971

        This function uses the below console commands that are made via gdal_translate.
        I will explain each and what they do along with how I have implemented them in python

        All time.sleep() are to allow time for the terminal thread to finish executing

        gdal_translate tiff_file intermediate_tif -b 3 -b 2 -b 1 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB
        This command takes out the first three bands of the original tiff file and moves them to correct band
        structure for an RBG image

        gdalinfo -mm intermediate_tif
        The next command is used to obtain the pixel min and max for each of the bands. The min and max for the
        entire file are then used in the below command.

        gdal_translate intermediate_tif scaled_tif -scale 570 23800 0 65535 -exponent 0.5 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB
        Now the tif file has its pixel values rescaled to values of a correct scale for showing color images.
        If this command is not run the image appears very dark.

        gdal_translate -of JPEG -scale -co worldfile=yes scaled_tif final_file_name
        Finally this last command translates the tif file into a JPEG file.


        :param tiff_file: tif file that needs to be translated into a JPEG
        :param intermediate_tif: name of the intermediate file in changing the given tif file to a JPEG image
        :param scaled_tif: scaled tif image
        :param final_file_name: name of the outputted JPEG file
        :return: numpy arrays that represent a JPEG image from the given tif file.
        """
        os.system("gdal_translate " + tiff_file + " " + intermediate_tif + " -b 3 -b 2 -b 1"
                                                                           " -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB")

        time.sleep(1)
        inter_tif = gdal.Open(intermediate_tif)
        stats = [inter_tif.GetRasterBand(i+1).ComputeStatistics(0) for i in range(inter_tif.RasterCount)]
        vmin, vmax, vmean, vstd = zip(*stats)
        pixel_max = max(vmax)
        pixel_min = max(vmin)

        os.system("gdal_translate " + intermediate_tif + ""
                                                         " " + scaled_tif + " -scale "
                                                                            + str(pixel_min) + " " + str(pixel_max)
                  + " 0 65535 -exponent 0.5 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB")
        time.sleep(1)

        os.system("gdal_translate -of JPEG -scale -co worldfile=yes " + scaled_tif + " " + final_file_name)
        time.sleep(1)
        return cv2.imread(final_file_name)





























