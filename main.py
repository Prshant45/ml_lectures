import cv2
from matplotlib import pyplot as plt
import numpy as np  

def print_image(image, title="hello", size=10):
    h, l = image.shape[0], image.shape[1]
    aspect_ratio =  h /l
    plt.figure(figsize=(size*aspect_ratio, size))
    # plt.imshow(image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   # plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
   # plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    plt.title(title)
    plt.show()


def main():
    image_data = cv2.imread("flower.jpg")
    building_data = cv2.imread("building.jpg")
    # print_image(image_data, "flower", 10)
    height, width = image_data.shape[0], image_data.shape[1]
    quater_height, quarter_width = height // 4, width // 4
    T = np.float32([[1, 0, quarter_width], [0, 1, quater_height]])
    translated_image = cv2.warpAffine(image_data, T, (100, 100))
    #print_image(translated_image, "flower", 10)

    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)
    translated_image = cv2.warpAffine(image_data, rotation_matrix, (width, height))
    #print_image(translated_image, "flower", 10)

    transposed_image = cv2.transpose(image_data)
    cv2.flip(transposed_image, 1)
    # print_image(transposed_image, "building", 10)

    resized_image = cv2.resize(image_data, (900, 400))
    # print_image(resized_image, "building", 10)

    resized_image = cv2.resize(image_data, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    # print_image(resized_image, "flower", 10)
    resized_image = cv2.resize(image_data, (1000, 1000), interpolation=cv2.INTER_LANCZOS4)
    # print_image(resized_image, "flower", 10)
    resized_image = cv2.resize(image_data, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    # print_image(resized_image, "flower", 10)

    resized_image = cv2.resize(image_data, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
    # print_image(resized_image, "flower", 10)
    resized_image = cv2.resize(image_data, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    # print_image(resized_image, "flower", 10)

    smaller = cv2.pyrDown(image_data)
    #print_image(smaller, "flower", 10)
    bigger = cv2.pyrUp(smaller)
    # print_image(bigger, "flower", 10)

    # Cropping
    cropped = image_data[100:400, 200:500]
    # print_image(cropped, "flower", 10)

    # drawing shapes
    copy_image = image_data.copy()
    cv2.rectangle(copy_image, (100,25), (410,350), (0,255,0), 3)
    #print_image(copy, "flower", 10)

    # cropped = image_data[100:400, 200:500]
    # print_image(cropped, "flower", 10)

    # Arthemtic operations
    M = np.ones(image_data.shape, dtype="uint8") * 100

    #addition 
    added = cv2.add(image_data, M)
    #print_image(added, "flower", 10)

    nrml_add = image_data + M
    #print_image(nrml_add, "flower", 10)

    sub = cv2.subtract(image_data, M)
    #print_image(sub, "flower", 10)

    nrml_substract = image_data - M
    # print_image(nrml_substract, "flower", 10)

    # bitwise operations
    xor_data = cv2.bitwise_xor(image_data, M)
    # print_image(xor_data, "flower", 10)

    not_data = cv2.bitwise_not(image_data)
    # print_image(not_data, "flower", 10)

    resized_flower_image = cv2.resize(image_data, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    resized_building_image = cv2.resize(building_data, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    add_image = cv2.add(resized_flower_image, resized_building_image)
    # print_image(add_image, "flower", 10)
    sub_image = cv2.subtract(add_image, resized_flower_image)
    # print_image(sub_image, "flower", 10)

    add_image = cv2.addWeighted(resized_flower_image, 1, resized_building_image, 1, 0)
    # print_image(add_image, "flower", 10)
    sub_image = cv2.subtract(add_image, resized_flower_image)
    # print_image(sub_image, "flower", 10)

    # blurring

    np_data = np.ones((3, 3), dtype="uint8") / 9
    blurred = cv2.filter2D(image_data, -1, np_data)
    # print_image(blurred, "flower", 10)
    blurred = cv2.GaussianBlur(image_data, (7, 7), 0)
    # print_image(blurred, "flower", 10)
    blurred = cv2.medianBlur(image_data, 7)
    # print_image(blurred, "flower", 10)
    blurred = cv2.bilateralFilter(image_data, 15, 75, 75)
    # print_image(blurred, "flower", 10)

    # noising
    # fastNlMeansDenoising - grayscale images
    # fastNlMeansDenoisingColored - colored images
    # fastNlMeansDenoisingMulti - for multiple grayscale images
    # fastNlMeansDenoisingColoredMulti - for multiple colored images
    
    #print_image(building_data, "flower", 10)
    de_noise = cv2.fastNlMeansDenoisingColored(building_data, None, 10, 10, 7, 21)
    #print_image(de_noise, "flower", 10)
    
    grey_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    #print_image(grey_image, "flower", 10)
    de_noise = cv2.fastNlMeansDenoising(grey_image, None, 10, 7, 21)
    #print_image(de_noise, "flower", 10)

    # sharpening
    print_image(image_data, "flower", 10)
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],   
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(image_data, -1, kernel_sharpening)
    # print_image(sharpened, "flower", 10)

    # Thresholding
    book_data = cv2.imread("book.png")
    # print_image(book_data, "book", 10)
    grey_image = cv2.cvtColor(book_data, cv2.COLOR_BGR2GRAY)
    # print_image(grey_image, "book_grey", 10)
    ret, thresh = cv2.threshold(grey_image, 127, 255, cv2.THRESH_BINARY)
    # print_image(thresh, "THRESH_BINARY", 10)
    ret, thresh = cv2.threshold(grey_image, 127, 255, cv2.THRESH_BINARY_INV)
    # print_image(thresh, "THRESH_BINARY_INV", 10)
    ret, thresh = cv2.threshold(grey_image, 127, 255, cv2.THRESH_TRUNC)
    # print_image(thresh, "THRESH_TRUNC", 10)
    ret, thresh = cv2.threshold(grey_image, 127, 255, cv2.THRESH_TOZERO)
    # print_image(thresh, "THRESH_TOZERO", 10)
    ret, thresh = cv2.threshold(grey_image, 127, 255, cv2.THRESH_TOZERO_INV)
    # print_image(thresh, "THRESH_TOZERO_INV", 10)
    thresh = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 10)
    # print_image(thresh, "ADAPTIVE_THRESH_MEAN_C", 10)
    thresh = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 10)
    # print_image(thresh, "ADAPTIVE_THRESH_GAUSSIAN_C", 10)

    from skimage.filters import threshold_local
    grey_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    v = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV))[2] # value channel
    # print_image(v[0], "v[0]", 10)
    T = threshold_local(grey_image, 11, offset=10, method="gaussian")
    thresh = (grey_image > T).astype("uint8") * 255
    # print_image(thresh, "thresh_skimage", 10)

    #   Dialation and Erosion

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    # print_image(dilated, "dilated", 10)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    # print_image(eroded, "eroded", 10)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # print_image(opening, "opening", 10)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # print_image(closing, "closing", 10)
    gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    # print_image(gradient, "gradient", 10)
    tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)
    # print_image(tophat, "tophat", 10)
    blackhat = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)
    # print_image(blackhat, "blackhat", 10)

    # Canny Edge Detection
    grey_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
   # print_image(grey_image, "grey", 10)
    inverted = cv2.bitwise_not(grey_image)
    #print_image(inverted, "inverted", 10)
    edges = cv2.Canny(inverted, 100, 200)
    # print_image(edges, "Canny_Edges", 10)
    # inverted_canny = cv2.bitwise_not(edges)
    # print_image(inverted_canny, "inverted_canny", 10)
    dilated = cv2.dilate(edges, kernel, iterations=1)
   # print_image(dilated, "dilated", 10)
    inverted_dilated = cv2.bitwise_not(dilated) 
    # print_image(inverted_dilated, "inverted_dilated", 10)

    def auto_canny(image, sigma=0.33):
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        v = np.median(blurred)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(blurred, lower, upper)
        return edged
    auto_edges = auto_canny(grey_image)
    print_image(auto_edges, "auto_canny", 10)
    inverted_auto_canny = cv2.bitwise_not(auto_edges)
    #print_image(inverted_auto_canny, "inverted_auto_canny", 10)


# contours
    contours, hierarchy = cv2.findContours(auto_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{len(contours)} contours found!")
    contour_image = image_data.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
    print_image(contour_image, "contours", 10)

    resized_flower_image = cv2.resize(image_data, (300, 300), interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(resized_flower_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{len(contours)} contours found!")
    contour_image = resized_flower_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
    print_image(contour_image, "contours", 10)


if __name__ == "__main__":
    main()




