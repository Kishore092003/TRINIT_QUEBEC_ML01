import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import os
# Initialize the Firebase application
cred = credentials.Certificate('C:/Users/kisho/Downloads/credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://object-detection-f7453-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Initialize the Firebase Realtime Database reference
db_ref = db.reference('annotations')

import xml.etree.ElementTree as ET

# Define the mapping of XML class labels to Firebase class labels
xml_label_map = {
    'D00':'Longitudinal Crack',
    'D10':'Transverse Crack' ,
    'D20':'Alligator Crack' ,
    'D40':'Pothole' ,
    'D44':'No Damage',
}

# Define the directory containing the XML files
xml_dir = "C:/Users/kisho/Downloads/RDD2022_India/India/train/annotations/jkl"

# Loop over the XML files in the directory
for xml_file in os.listdir(xml_dir):
    # Parse the XML file
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()

    # Loop over the objects in the XML file
    for obj in root.iter('object'):
        # Extract the bounding box coordinates and class label
        bndbox = obj.find('bndbox')
        x, y, w, h = map(float, [bndbox.find('xmin').text, bndbox.find('ymin').text, bndbox.find('xmax').text, bndbox.find('ymax').text])
        try:
            label = xml_label_map[obj.find('name').text]
        except KeyError:
            print(f"Unknown class label: {obj.find('name').text}")
            xml_label_map[obj.find('name').text]="No Damage"
            continue

        # Convert the bounding box coordinates to Firebase format
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

        # Store the annotation in Firebase
        db_ref.push({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'label': label
        })
