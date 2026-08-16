import os
import json
import xml.etree.ElementTree as ET
from collections import Counter

def explore_coco_json(json_path):
    print(f"\n--- Exploring COCO JSON: {os.path.basename(json_path)} ---")
    if not os.path.exists(json_path):
        print("File does not exist.")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("Keys in JSON:", list(data.keys()))
    if 'images' in data:
        print("Total images:", len(data['images']))
        print("Sample image object:", data['images'][0] if len(data['images']) > 0 else 'None')
    if 'annotations' in data:
        print("Total annotations:", len(data['annotations']))
        print("Sample annotation:", data['annotations'][0] if len(data['annotations']) > 0 else 'None')
    if 'categories' in data:
        print("Categories:", data['categories'])

def explore_xml_annotations(xml_dir):
    print(f"\n--- Exploring XML Annotations in {os.path.basename(xml_dir)} ---")
    if not os.path.exists(xml_dir):
        print("Directory does not exist.")
        return
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} XML files.")
    
    if len(xml_files) == 0:
        return
    
    # Parse a sample XML
    sample_xml = os.path.join(xml_dir, xml_files[0])
    print(f"Sample XML file: {xml_files[0]}")
    tree = ET.parse(sample_xml)
    root = tree.getroot()
    
    print("Root tag:", root.tag)
    for child in root:
        if child.tag == 'size':
            print(f"Image Size: width={child.find('width').text}, height={child.find('height').text}, depth={child.find('depth').text}")
        elif child.tag == 'object':
            name = child.find('name').text
            obj_type = child.find('type').text if child.find('type') is not None else 'unknown'
            print(f"Object: name='{name}', type='{obj_type}'")
            if child.find('bndbox') is not None:
                bbox = child.find('bndbox')
                print(f"  Standard Bbox: xmin={bbox.find('xmin').text}, ymin={bbox.find('ymin').text}, xmax={bbox.find('xmax').text}, ymax={bbox.find('ymax').text}")
            if child.find('robndbox') is not None:
                robox = child.find('robndbox')
                print(f"  Rotated Bbox: cx={robox.find('cx').text}, cy={robox.find('cy').text}, w={robox.find('w').text}, h={robox.find('h').text}, angle={robox.find('angle').text}")

def explore_dataset_structure(base_dir):
    print(f"\n--- Exploring Dataset Directory Structure: {base_dir} ---")
    sublevels = os.path.join(base_dir, "Sub-levels")
    if not os.path.exists(sublevels):
        print("Sub-levels directory not found.")
        return
    
    for path, dirs, files in os.walk(sublevels):
        depth = path.replace(sublevels, '').count(os.sep)
        if depth <= 3:
            jpgs = [f for f in files if f.lower().endswith('.jpg')]
            xmls = [f for f in files if f.lower().endswith('.xml')]
            h5s = [f for f in files if f.lower().endswith('.h5')]
            pkls = [f for f in files if f.lower().endswith('.pkl')]
            if jpgs or xmls or h5s or pkls or dirs:
                indent = '  ' * depth
                print(f"{indent}- {os.path.basename(path)}/ (dirs: {len(dirs)}, jpgs: {len(jpgs)}, xmls: {len(xmls)}, h5s: {len(h5s)}, pkls: {len(pkls)})")

if __name__ == "__main__":
    base_bristol = r"c:\Users\NISARG\BAIF\data\bristol\dataset_bristol"
    
    # 1. Explore directory structure
    explore_dataset_structure(base_bristol)
    
    # 2. Explore COCO JSON
    train_json = os.path.join(base_bristol, "Sub-levels", "Detection_and_localisation", "Train", "annotations", "instances_train.json")
    val_json = os.path.join(base_bristol, "Sub-levels", "Detection_and_localisation", "Train", "annotations", "instances_val.json")
    explore_coco_json(train_json)
    explore_coco_json(val_json)
    
    # 3. Explore XML annotations
    test_xml_dir = os.path.join(base_bristol, "Sub-levels", "Detection_and_localisation", "Test", "images", "val")
    explore_xml_annotations(test_xml_dir)
