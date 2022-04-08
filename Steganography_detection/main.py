import cv2
import glob
import re
import os
import numpy as np
import json
import csv


def grayscale_and_resize_images():
    labels = ['Cover', 'JMiPOD', 'JUNIWARD', 'Test', 'UERD']
    data_dir = r'D:\Repo_Licenta\alaska2-steganalysis'
    put_dir = r'D:\Repo_Licenta\resized_grayscale_alaska2'
    img_size = 128

    for label in labels:
        path = os.path.join(data_dir, label)
    path = os.path.join(path, '*.jpg')
    put_path = os.path.join(put_dir, label)
    for filename in glob.glob(path):
        print(filename)
    img = cv2.imread(filename)
    rl = cv2.resize(img, (img_size, img_size))
    gray_image = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
    img_name = re.split(r'\\', filename)[-1]
    destination_dir = os.path.join(put_path, img_name)
    cv2.imwrite(destination_dir, gray_image)


def extractimgarray_normalize_writecsv():
    labels = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    data_dir = r'D:\Repo_Licenta\resized_grayscale_alaska2'
    img_size = 128
    header = ['img', 'label']
    with open('data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    img_arr = np.array(img_arr) / 255
                    img_arr = img_arr[..., np.newaxis]
                    img_arr.reshape(img_size, img_size, 1)
                    writer.writerow([img_arr, class_num])
                except Exception as e:
                    print(e)


