"""
ECE276A WI20 HW1
Stop Sign Detector
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


class StopSignDetector():
    def __init__(self):
        """
            Initialize your stop sign detector with the attributes you need,
            e.g., parameters of your classifier
        """
        self.w = np.load("weights_MLE.npy")

    def segment_image(self, img):
        """
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        """
        # YOUR CODE HERE

        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = img / 255.0

        shp = img.shape
        X = np.reshape(img, (shp[0] * shp[1], 3))
        one = np.ones((X.shape[0], 1))
        X = np.concatenate((one, X), 1)

        # Load the weights
        w = self.w
        y_pd = np.matmul(X, w) >= 0
        y_pd = y_pd.astype(np.uint8)
        mask_img = np.reshape(y_pd, (shp[0], shp[1]))
        return mask_img

    def get_bounding_box(self, img):
        """
            Find the bounding box of the stop sign
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        """
        # YOUR CODE HERE

        mask_img = self.segment_image(img)

        kernel = np.ones((3, 3), np.uint8)
        dil_mask = cv2.dilate(mask_img, kernel, iterations=2)

        cont, h = cv2.findContours(dil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h = h[0]
        bbox = []
        max_h = mask_img.shape[0]
        max_w = mask_img.shape[1]
        img_area = max_h*max_w

        for c in zip(cont, h):
            c_cont = c[0]
            thresh = 0.01 * cv2.arcLength(c_cont, True)
            poly_aprox = cv2.approxPolyDP(c_cont, thresh, True)
            
            x1, y1, w, h = cv2.boundingRect(c_cont)
            aspr = w / h
            bb_area = w * h
            if aspr < 1.2 and aspr > 0.8:
                if bb_area / img_area > 0.008:
                    bbox.append([x1, max_h - (y1 + h), x1+w, max_h - y1])
        print(len(bbox), bbox)
        return bbox


if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()
    for filename in os.listdir(folder):
        # read one test image
        #filename = "38.jpg"
        img = cv2.imread(os.path.join(folder, filename))
        print(img.shape, img.dtype, np.min(img), np.max(img))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display results:
        # (1) Segmented images
        mask_img = my_detector.segment_image(img)

        # (2) Stop sign bounding box
        boxes = my_detector.get_bounding_box(img)
        # The autograder checks your answers to the functions segment_image() and get_bounding_box()
        # Make sure your code runs as expected on the testset before submitting to Gradescope