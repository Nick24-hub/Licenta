import cv2
import glob
import re
import os
import numpy as np
import json
import csv


def grayscale_and_resize_images_alaska():
    labels = ['Cover', 'JMiPOD', 'JUNIWARD', 'Test', 'UERD']
    data_dir = r'D:\Repo_Licenta\alaska2-steganalysis'
    put_dir = r'D:\Repo_Licenta\resized_grayscale_alaska2'
    img_size = 128

    for label in labels:
        path = os.path.join(data_dir, label)
        path = os.path.join(path, '*.jpg')
        put_path = os.path.join(put_dir, label)
        for filename in glob.glob(path):
            # print(filename)
            img = cv2.imread(filename)
            resized_img = cv2.resize(img, (img_size, img_size))
            gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            img_name = re.split(r'\\', filename)[-1]
            destination_dir = os.path.join(put_path, img_name)
            cv2.imwrite(destination_dir, gray_image)


def grayscale_and_resize_images_digital():
    labels = ['dct', 'fft', 'lsb_grayscale', 'ssb4', 'ssbn']
    data_dir = r'D:\Repo_Licenta\Digital_Steganography'
    put_dir = r'D:\Repo_Licenta\resized_grayscale_digital_steganography'
    img_size = 128

    for label in labels:
        path = os.path.join(data_dir, label)
        if (label == 'dct' or label == 'fft'):
            path = os.path.join(path, '*.bmp')
        else:
            path = os.path.join(path, '*.png')
        put_path = os.path.join(put_dir, label)
        for filename in glob.glob(path):
            # print(filename)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (img_size, img_size))
            img_name = re.split(r'\\', filename)[-1]
            destination_dir = os.path.join(put_path, img_name)
            cv2.imwrite(destination_dir, resized_img)


grayscale_and_resize_images_digital()


def extract_img_array_normalize_writecsv():
    labels = ['Cover', 'JMiPOD', 'JUNIWARD', 'Test', 'UERD']
    data_dir = r'D:\Repo_Licenta\resized_grayscale_alaska2'
    put_dir = r'D:\Repo_Licenta\CSV'
    img_size = 128

    for label in labels:
        path = os.path.join(data_dir, label)
        path = os.path.join(path, '*.jpg')
        put_path = os.path.join(put_dir, label)
        for filename in glob.glob(path):
            # print(filename)
            try:
                img_arr = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                img_arr = np.array(img_arr) / 255
                # img_arr = img_arr[..., np.newaxis]
                # img_arr.reshape(img_size, img_size, 1)
                img_name = re.split(r'\\', filename)[-1]
                img_name = re.split(r'.jpg', img_name)[0]
                img_name = img_name + '.csv'
                destination_dir = os.path.join(put_path, img_name)
                np.savetxt(destination_dir, img_arr, delimiter=',')
            except Exception as e:
                print(e)
