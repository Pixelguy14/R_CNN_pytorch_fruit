import csv
import os
import xml.etree.ElementTree as ET
import ast
from PIL import Image

# Paths
csv_file = 'dataset/annotations/dataset_10_each_csv.csv'
output_dir = 'dataset/annotations/voc_annotations'
image_dir = 'dataset/images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_voc_annotation(filename, annotations, output_dir, image_dir):
    annotation = ET.Element('annotation')
    
    folder = ET.SubElement(annotation, 'folder').text = os.path.basename(image_dir)
    ET.SubElement(annotation, 'filename').text = filename
    ET.SubElement(annotation, 'path').text = os.path.join(image_dir, filename)
    
    # Read the actual image dimensions
    img_path = os.path.join(image_dir, filename)
    with Image.open(img_path) as img:
        width, height = img.size
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    
    for ann in annotations:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = ann['category']
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(ann['x'])
        ET.SubElement(bbox, 'ymin').text = str(ann['y'])
        ET.SubElement(bbox, 'xmax').text = str(ann['x'] + ann['width'])
        ET.SubElement(bbox, 'ymax').text = str(ann['y'] + ann['height'])
    
    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(output_dir, os.path.splitext(filename)[0] + '.xml'))

annotations_dict = {}

with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        filename = row['filename']
        shape_attributes = ast.literal_eval(row['region_shape_attributes'])
        region_attributes = ast.literal_eval(row['region_attributes'])
        if filename not in annotations_dict:
            annotations_dict[filename] = []
        annotations_dict[filename].append({
            'width': shape_attributes['width'],
            'height': shape_attributes['height'],
            'x': shape_attributes['x'],
            'y': shape_attributes['y'],
            'category': region_attributes['fruta']
        })

for filename, annotations in annotations_dict.items():
    create_voc_annotation(filename, annotations, output_dir, image_dir)

print("Conversion to VOC format completed.")
