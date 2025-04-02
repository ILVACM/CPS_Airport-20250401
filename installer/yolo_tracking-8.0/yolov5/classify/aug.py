import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
import glob

def rotate_image_and_boxes(image, boxes, angle):
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += new_width / 2 - image_center[0]
    rotation_mat[1, 2] += new_height / 2 - image_center[1]

    rotated_image = cv2.warpAffine(image, rotation_mat, (new_width, new_height))

    new_boxes = []
    for box in boxes:
        new_box = []
        for point in box:
            x = point[0]
            y = point[1]
            new_x = abs_cos * x - abs_sin * y + rotation_mat[0, 2]
            new_y = abs_sin * x + abs_cos * y + rotation_mat[1, 2]
            new_box.append((int(new_x), int(new_y)))
        new_boxes.append(new_box)

    return rotated_image, new_boxes

def flip_image_and_boxes(image, boxes):
    flipped_image = cv2.flip(image, 1)
    width = image.shape[1]
    new_boxes = []
    for box in boxes:
        new_box = []
        for point in box:
            new_x = width - point[0]
            new_y = point[1]
            new_box.append((new_x, new_y))
        new_boxes.append(new_box)

    return flipped_image, new_boxes

def resize_image_and_boxes(image, boxes, scale):
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    new_boxes = []
    for box in boxes:
        new_box = []
        for point in box:
            new_x = int(point[0] * scale)
            new_y = int(point[1] * scale)
            new_box.append((new_x, new_y))
        new_boxes.append(new_box)

    return resized_image, new_boxes

def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for object in root.findall('object'):
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    return boxes

def save_new_image_and_annotation(image, boxes, image_path, xml_path, suffix, save_folder):
    new_image_name = os.path.basename(os.path.splitext(image_path)[0]) + '_' + suffix + '.jpg'
    new_image_path = os.path.join(save_folder, new_image_name)
    cv2.imwrite(new_image_path, image)

    new_xml_name = os.path.basename(os.path.splitext(xml_path)[0]) + '_' + suffix + '.xml'
    new_xml_path = os.path.join(save_folder, new_xml_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for i, object in enumerate(root.findall('object')):
        bndbox = object.find('bndbox')
        box = boxes[i]
        bndbox.find('xmin').text = str(min(point[0] for point in box))
        bndbox.find('ymin').text = str(min(point[1] for point in box))
        bndbox.find('xmax').text = str(max(point[0] for point in box))
        bndbox.find('ymax').text = str(max(point[1] for point in box))
    tree.write(new_xml_path)

def process_all_images_and_annotations(input_folder, save_folder):
    xml_files = glob.glob(os.path.join(input_folder, '*.xml'))
    for xml_path in xml_files:
        image_path = os.path.splitext(xml_path)[0] + '.jpg'
        if not os.path.exists(image_path):
            print(f"No image found for XML file {xml_path}")
            continue
        
        image = cv2.imread(image_path)
        boxes = parse_annotation(xml_path)

        rotated_image, rotated_boxes = rotate_image_and_boxes(image, boxes, 45)
        save_new_image_and_annotation(rotated_image, rotated_boxes, image_path, xml_path, 'rotated', save_folder)

        flipped_image, flipped_boxes = flip_image_and_boxes(image, boxes)
        save_new_image_and_annotation(flipped_image, flipped_boxes, image_path, xml_path, 'flipped', save_folder)

        resized_image, resized_boxes = resize_image_and_boxes(image, boxes, 0.5)
        save_new_image_and_annotation(resized_image, resized_boxes, image_path, xml_path, 'resized', save_folder)

def main(input_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    process_all_images_and_annotations(input_folder, save_folder)

if __name__ == "__main__":
    # 请替换 'your_input_folder' 为你的图像和 XML 文件的文件夹路径
    # 替换 'your_save_folder' 为你想保存增强后的图像和 XML 文件的文件夹路径
    main('/media/kevin/B044E2E97FC50881/Prohibited/dataset/Selected/26', '/media/kevin/B044E2E97FC50881/Prohibited/dataset/Selected/testzdz')
