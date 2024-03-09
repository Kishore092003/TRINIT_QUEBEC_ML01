import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("C:/Users/kisho/Downloads/credentials.json")
firebase_admin.initialize_app(cred)

import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import torch

# Initialize the YOLOv8s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the Firebase application
cred = credentials.Certificate('C:/Users/kisho/Downloads/credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://object-detection-f7453-default-rtdb.asia-southeast1.firebasedatabase.app/annotations'
})

# Retrieve the annotations from Firebase
annotations_ref = db.reference('annotations')
annotations = annotations_ref.get()

# Load an input image
img = cv2.imread('"C:/Users/kisho/Downloads/RDD2022_India/India/test/images/India_009862.jpg"')

# Perform object detection on the input image
results = model(img)

# Draw bounding boxes and labels on the input image
for box in results.xyxy[0]:
    x1, y1, x2, y2, label, confidence = box
    x1, y1, x2, y2 = map(int, box[:4])
    label_name = (labels[int(label)] if labels else label)
    color = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Overlay the annotations on the input image
for annotation in annotations:
    annotation_bbox = annotation['bbox']
    x1, y1, x2, y2 = annotation_bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    color = (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, annotation['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the output image
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
