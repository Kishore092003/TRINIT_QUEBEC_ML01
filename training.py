import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

# Function to calculate Intersection over Union (IoU)
def calculate_iou(pred_box, true_box):
    x1, y1, w1, h1 = pred_box
    x2, y2, w2, h2 = true_box

    intersection_x = max(x1, x2)
    intersection_y = max(y1, y2)
    intersection_w = min(x1 + w1, x2 + w2) - intersection_x
    intersection_h = min(y1 + h1, y2 + h2) - intersection_y

    if intersection_w <= 0 or intersection_h <= 0:
        return 0

    intersection_area = intersection_w * intersection_h
    pred_box_area = w1 * h1
    true_box_area = w2 * h2

    iou = intersection_area / (pred_box_area + true_box_area - intersection_area)
    return iou

# Function to calculate the localization loss
def calculate_bbox_loss(pred_box, true_box):
    pred_x, pred_y, pred_w, pred_h = pred_box
    true_x, true_y, true_w, true_h = true_box

    x_diff = (true_x - pred_x) * 0.5
    y_diff = (true_y - pred_y) * 0.5
    w_diff = torch.log(true_w / pred_w)
    h_diff = torch.log(true_h / pred_h)

    loss = x_diff ** 2 + y_diff ** 2 + w_diff ** 2 + h_diff ** 2
    return loss

# Function to load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
    with open("D00 Longitudinal Crack.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

# Function to preprocess images and annotations
def preprocess_data(image, input_size):
    height, width, _ = image.shape
    resized_image = cv2.resize(image, (input_size, input_size))
    preprocessed_image = cv2.dnn.blobFromImage(resized_image, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False)
    return preprocessed_image

# Function to load training data from Firebase
def load_training_data():
    training_data = []
    folder_path = "train/images"
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    annotation_folder_path = "train/labels"
    for image_path in image_paths:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        annotation_file_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        with open(os.path.join(annotation_folder_path, annotation_file_name), 'r') as f:
            annotations = [line.strip().split(' ') for line in f.readlines()]
            annotations = [(int(float(a[0])), int(float(a[1])), int(float(a[2])), int(float(a[3]))) for a in annotations]
            training_data.append((image, annotations))
    return training_data


