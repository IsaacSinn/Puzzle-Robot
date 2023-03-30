#!/usr/bin/env python
"""This is a library of functions for performing color-based image segmentation of an image."""

import cv2
import numpy as np


def image_one_to_three_channels(img):
    """ Transforms an image from two channels to three channels """
    #First promote array to three dimensions, then repeat along third dimension
    img_three = np.tile(img.reshape(img.shape[0], img.shape[1], 1), (1, 1, 3))
    return img_three


def classifier_parameters():
    lb = (125, 125, 125)
    ub = (255, 255, 255)
    return lb, ub


def pixel_count(img):
    nb_positive = np.count_nonzero(img)
    nb_negative = img.size - nb_positive
    return nb_positive, nb_negative


def pixel_count_segmentation(filename):
    lb, ub = classifier_parameters()

    img = cv2.imread(filename)
    img_seg = cv2.inRange(img, lb, ub)
    nb_positive, nb_negative = pixel_count(img_seg)
    return img_seg.size, nb_positive, nb_negative


def precision_recall(true_positive, false_positive, false_negative):
    recall = float(true_positive) / (true_positive + false_negative)
    total_positive = true_positive + false_positive
    if total_positive == 0:
        precision = 1
    else:
        precision = float(true_positive) / total_positive
    return precision, recall


def segmentation_statistics(filename_positive, filename_negative):
    total_positive, true_positive, false_negative = pixel_count_segmentation(
        filename_positive)
    total_negative, false_positive, true_negative = pixel_count_segmentation(
        filename_negative)

    precision, recall = precision_recall(true_positive, false_positive,
                                         false_negative)

    print ("Nb. of positive examples:", total_positive)
    print ("Nb. of negative examples:", total_negative)
    print ("True positives:", true_positive)
    print ("False positives:", false_positive)
    print ("False negatives:", false_negative)
    print ("True negatives:", true_negative)
    print ("Precision:", precision)
    print ("Recall:", recall)

    return precision, recall


def image_centroid_horizontal(img):
    # img contains only black and white pixels
    total_x = np.array(np.where(img == 255))
    total_x = total_x[1, :]
    if len(total_x) == 0:
        x_centroid = 0
    else:
        x_centroid = int(np.median(total_x))
    return x_centroid


def image_line_vertical(img, x):
    """ Adds a green 3px vertical line to the image """
    cv2.line(img, (x, 0), (x, img.shape[1]), (0, 255, 0), 3)
    return img
