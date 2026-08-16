import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def crop_and_rectify(image_path, xml_path, output_dir):
    if not os.path.exists(image_path) or not os.path.exists(xml_path):
        return False
    
    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return False
    
    # Find robndbox parameters
    robndbox = None
    for obj in root.findall('object'):
        if obj.find('type') is not None and obj.find('type').text == 'robndbox':
            robndbox = obj.find('robndbox')
            break
            
    if robndbox is None:
        return False
    
    try:
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)  # in radians
    except Exception as e:
        print(f"Error extracting robndbox parameters from {xml_path}: {e}")
        return False
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image {image_path}")
        return False
    
    img_h, img_w, _ = img.shape
    
    # Convert angle from radians to degrees
    angle_deg = np.degrees(angle)
    
    # Get rotation matrix (rotate around bbox center in the opposite direction of the angle to align it horizontally)
    # OpenCV rotates counter-clockwise for positive degrees
    # If the bbox angle is positive (clockwise or counter-clockwise), we rotate by -angle_deg (in degrees)
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    
    # Rotate the entire image
    rotated = cv2.warpAffine(img, M, (img_w, img_h))
    
    # Crop the horizontal bbox
    # Center-aligned crop of size (w, h)
    xmin = int(cx - w / 2)
    xmax = int(cx + w / 2)
    ymin = int(cy - h / 2)
    ymax = int(cy + h / 2)
    
    # Clip coordinates to image boundary
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)
    
    cropped = rotated[ymin:ymax, xmin:xmax]
    
    if cropped.size == 0:
        return False
    
    # Save cropped image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
    cv2.imwrite(out_path, cropped)
    return True

def process_all_val_images(val_dir, output_dir, max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(val_dir) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} xml annotations in {val_dir}.")
    
    processed_count = 0
    skipped_count = 0
    
    for i, xml_file in enumerate(xml_files):
        if max_images is not None and processed_count >= max_images:
            break
            
        base_name = os.path.splitext(xml_file)[0]
        image_file = f"{base_name}.jpg"
        
        image_path = os.path.join(val_dir, image_file)
        xml_path = os.path.join(val_dir, xml_file)
        
        success = crop_and_rectify(image_path, xml_path, output_dir)
        if success:
            processed_count += 1
        else:
            skipped_count += 1
            
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(xml_files)} files...")
            
    print(f"\nProcessing complete! Successfully cropped {processed_count} cows, skipped {skipped_count}.")

if __name__ == "__main__":
    val_dir = r"c:\Users\NISARG\BAIF\data\bristol\dataset_bristol\Sub-levels\Detection_and_localisation\Test\images\val"
    output_dir = r"c:\Users\NISARG\BAIF\data\processed\bristol_cropped"
    
    # Crop all validation images (useful for our downstream analysis)
    process_all_val_images(val_dir, output_dir)
