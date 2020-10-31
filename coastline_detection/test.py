from skimage import io, img_as_ubyte
import numpy as np
import cv2
import matplotlib.pyplot as plt


def blend_images(im_1, im_2, alpha=0.7, name="blended.png"):
    src1 = cv2.imread(im_1)
    src2 = cv2.imread(im_2)
    print(src1.shape)
    print(src2.shape)

    # for lack of a better way to solve this
    for x in range(src2.shape[0]):
        for y in range(src2.shape[1]):
            src2[x, y] = [0, src2[x, y, 0], src2[x, y, 0]]

    beta = (1.0 - alpha)
    dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
    cv2.imwrite("edge_results/blend/" + name, dst)


def run_a_bunch_of_samples(im, blur, upper_bounds, lower_bounds, directory):
    im = cv2.GaussianBlur(im, blur, 0)

    row_count = 0
    col_count = 0
    for col in upper_bounds:
        for row in lower_bounds[col_count]:
            canny = cv2.Canny(im, row, col)
            print("Col " + str(col_count) + "    row " + str(row_count))
            cv2.imwrite(directory + "/Upper " + str(col) + " , lower " + str(row) + ".png", canny)
            row_count += 1
        row_count = 0
        col_count += 1


def grayscale_with_color_bar(image, colormap="gist_gray", name="grayscale with colorbar.png"):
    plt.imshow(image, cmap=colormap)
    plt.colorbar()
    plt.savefig("grayscale_colorbar/" +name)
    plt.show()


def sobel(image, name="sobel.png"):
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    cv2.imwrite("edge_results/sobel" + name, sobelCombined)




"""

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"
             "edge_results/good_canny_edge/Upper 50 , lower 20_output.png",
             name="out_canny_20:50.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/edge_results/holistic/output.png",
             name="out_holistic.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output.jpg",
              "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/edge_results/isodata/Canny_isodata2.png",
              name="out_isodata.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"
             "edge_results/ndwi/uint8_selectorndwi_image2.png",
             name="out_ndwi.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/edge_results/"
             "ndwi_edge/out_ndwi_canny_edge.png",
             name="out_ndwi_canny_edge.png")

##############################################################################################################
blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"
             "edge_results/good_canny_edge/Upper 110 , lower 70_output2.png",
             name="out2_canny_70:110.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"
             "edge_results/good_canny_edge/Upper 160 , lower 70_output2.png",
             name="out2_canny_70:160.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/edge_results/holistic/output2.png",
             name="out2_holistic.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/edge_results/isodata/Canny_isodata.png",
              name="out2_isodata.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"
             "edge_results/ndwi/uint8_selectorndwi_image.png",
             name="out2_ndwi.png")

blend_images("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg",
             "/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/edge_results/"
             "ndwi_edge/out2_ndwi_canny_edge.png",
             name="out2_ndwi_canny_edge.png")



"""

"""
ndwi_image = cv2.imread("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"
                        "testing canny uint8 ndwi/Upper 200 , lower 90.png")

rgb_image = cv2.imread("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/"
                       "edge testing cv uint8 rbg_converted_int8/Upper 80 , lower 20.png")
n = ndwi_image[:, :, 0]
r = rgb_image[:, :, 1]
for line in ndwi_image[:, :, 0]:
    print(line)
print(ndwi_image[:, :, 0])

n = np.uint8(n > 1)
r = np.uint8(r > 1)

cv2.imshow("ndwi", n)
cv2.imshow("rgb", r)

vis = ndwi_image + rgb_image

cv2.imshow("combined", vis)




cv2.waitKey(0)

"""




"""
image = cv2.imread("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/uint8_ndwi_image.png")
mask = cv2.imread("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/uint8_selectorndwi_image.png", 0)
cv2.imshow("mask", mask)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("image", image)

m = cv2.bitwise_not(image, image, mask=mask)
cv2.imshow("masked", m)

image = cv2.GaussianBlur(image, (5, 5), 0)
# cv2.imshow("Blurred", image)
cv2.imwrite("saved_outputs_opencv/Blurred.png", image)
canny = cv2.Canny(image, 20, 150)
cv2.imshow("Canny", canny)
cv2.imwrite("saved_outputs_opencv/Canny Edge.png", canny)
cv2.waitKey(0)

"""


"""
# testing for ndwi image

image = cv2.imread("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
u = []
l = []

for i in range(20, 210, 10):
    u.append(i)
    temp = []
    for j in range(0, i, 10):
        temp.append(j)
    l.append(temp)

run_a_bunch_of_samples(image, (5, 5), u, l, "testing canny uint8 ndwi")

"""

"""


First set of sample runs

image = cv2.imread("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/rgb_converted_int8.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
upper = [80, 70, 60, 50]
lower = [[20, 30, 40, 50, 60], [20, 30, 40, 50, 60], [10, 20, 30, 40, 50], [0, 10, 20, 30, 40]]
run_a_bunch_of_samples(image, (5, 5,), upper, lower, "edge testing cv uint8 rbg_converted_int8")

"""


"""
# trial run testing code, colorbar with matplotlib, canny edge with opencv


image = cv2.imread("/home/nelson/PycharmProjects/coastline_detection_with_LANDSAT/good_images/output2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image, cmap="gist_gray")
plt.colorbar()
plt.savefig("grayscale with colorbar.png")
plt.show()

# cv2.imshow("test", image)
cv2.imwrite("saved_outputs_opencv/Original.png", image)

image = cv2.GaussianBlur(image, (5, 5), 0)
# cv2.imshow("Blurred", image)
cv2.imwrite("saved_outputs_opencv/Blurred.png", image)
canny = cv2.Canny(image, 20, 50)
# cv2.imshow("Canny", canny)
cv2.imwrite("saved_outputs_opencv/Canny Edge.png", canny)
cv2.waitKey(0)


"""





