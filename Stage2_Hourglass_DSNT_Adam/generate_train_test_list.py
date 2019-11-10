import numpy as np

import os
import cv2
import random
import matplotlib.pyplot as plt
from random import sample
import torch
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


data_folder = '../data'
folder_list = ['I', 'II', 'total_data']
img_folder_I = os.path.join(data_folder, folder_list[0])
img_folder_II = os.path.join(data_folder, folder_list[1])
img_folder_total = os.path.join(data_folder, folder_list[2])


# view the data and labels
def random_view_data_and_label(img_folder, expand=False, ratio=4, cropped=False, txt_name='/label.txt'):
    """
    img_folder: folder path which contains the image data and labels txt

    randomly choose a image from the folder and show the corresponding labels and bounding box.
    """
    print("label txt is from path:", img_folder + txt_name)
    with open(img_folder + txt_name) as labels:
        lines = labels.readlines()
    random_line = random.choice(lines)
    # print(random_line)
    split_line = random_line.split()
    # print('split_line', split_line)
    image_path = os.path.join(img_folder, split_line[0])
    random_img = cv2.imread(image_path)
    print("Viewing the image from path: ", image_path)
    # print('random_img', random_img)
    print('The shape of origin image:', random_img.shape)
    bbox = list(map(float, split_line[1:5]))
    bbox = list(map(int, bbox))
    print('bbox', bbox)
    if expand:
        bbox = expand_roi(bbox, random_img.shape, ratio=ratio)
    cv2.rectangle(random_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    if cropped:
        print("The image has been cropped and only containing a face.")
        random_img = random_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    keypoints = list(map(float, split_line[5:]))
    keypoints = list(map(int, keypoints))
    # print('keypoints', keypoints)
    # print(len(keypoints))
    for i in range(0, len(keypoints), 2):
        keypoint_w, keypoint_h = int(keypoints[i]), int(keypoints[i + 1])
        cv2.circle(random_img, (keypoint_w, keypoint_h), 2, (0, 0, 255), -1)
    cv2.imshow(random_line.split()[0], random_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def expand_roi(bbox, img_shape, ratio=4):
    """
    bbox: a list, [x1, y1, x2, y2
    img_shape: shape of image, (H, W, C)
    ratio: expand ratio.
    Expand the bbox with the ratio so that the bbox can contain all key points in .
    """
    w1, h1, w2, h2 = bbox
    h_max, w_max = img_shape[0] - 1, img_shape[1] - 1
    h_min, w_min = 0, 0
    ww1, hh1 = w1 - (w2 - w1) // ratio, h1 - (h2 - h1) // ratio
    ww2, hh2 = w2 + (w2 - w1) // ratio, h2 + (h2 - h1) // ratio
    ww1 = min(max(w_min, ww1), w_max)
    hh1 = min(max(h_min, hh1), h_max)
    ww2 = min(max(w_min, ww2), w_max)
    hh2 = min(max(h_min, hh2), h_max)
    return ww1, hh1, ww2, hh2


def remove_invalid_image(lines):
    """
    Eliminate invalid image.
    """
    images = []
    for line in lines:
        name = line.split()[0]
        if os.path.isfile(name):
            images.append(line)
    return images


def load_metadata(data_fold, folder_name):
    """
    data_fold: fold contains all data.
    folder_name: fold contatins the images and label txt.
    return: a list of labels,[[file_path, x1, y1, x2, y2, keypoint1_x, keypoint1_y, ..., keypoint21_x, keypoint21_y],..]
    """
    raw_lines = []
    folder = os.path.join(data_fold, folder_name)
    label_file = os.path.join(folder, 'label.txt')
    with open(label_file) as labels:
        lines = labels.readlines()
    raw_lines.extend(list(map((folder + '/').__add__, lines)))
    res_lines = remove_invalid_image(raw_lines)
    return res_lines


def process_res_lines(res_lines: list):
    """
    Process the result lines, expand the x1, y1, x2, y2, and subtract x1, y1 from keypoint x and y.
    res_lines:
    a list of labels,[[file_path, x1, y1, x2, y2, keypoint1_x, keypoint1_y, ..., keypoint21_x, keypoint21_y],..]
    return:
    a list of labels,[[file_path, x1, y1, x2, y2, keypoint1_x, keypoint1_y, ..., keypoint21_x, keypoint21_y],..]
    """
    print("Processing the labels...")
    for res_line in res_lines:
        img = cv2.imread(res_line[0])
        res_line[1:5] = expand_roi(list(map(float, res_line[1:5])), img.shape)
        res_line[5:] = list(map(float, res_line[5:]))
        res_line[5::2] = list(np.array(res_line[5::2]) - res_line[1])
        res_line[6::2] = list(np.array(res_line[6::2]) - res_line[2])
    print("Total labels processed.")
    # res_in_nparray = np.array(res_lines)
    # print(res_in_nparray[0])
    # for i in range(4, res_in_nparray.shape[1], 2):
    #    res_in_nparray[:, i:i + 2] -= res_in_nparray[:, 1:3]
    # print(res_in_nparray[0])
    return res_lines


def output_new_labels(res_lines, check_txt=False):
    """
    Shuffle the labels, and output the new labels into train.txt, valid.txt, test.txt. Ratio is 8: 1: 1.
    """
    print("Origin labels first 3 lines: ", res_lines[0:2])
    random.shuffle(res_lines)
    print("Shuffled labels first 3 lines: ", res_lines[0:2])

    label_amount = len(res_lines)
    train_num = int(label_amount * 0.8)
    test_num = int(label_amount * 0.1)
    print("Number of total labels:", label_amount)
    print("Number of train labels:", train_num)
    print("Number of test labels:", test_num)
    with open("../data/train.txt", 'w') as train:
        print("Generate train.txt...")
        for train_label in res_lines[: train_num]:
            train_label = list(map(str, train_label))
            train.write(' '.join(train_label) + '\n')
        print("train.txt generated.")
    with open("../data/valid.txt", 'w') as valid:
        print("Generate valid.txt...")
        for valid_label in res_lines[train_num:-test_num]:
            valid_label = list(map(str, valid_label))
            valid.write(' '.join(valid_label) + '\n')
        print("valid.txt generated.")
    with open("../data/test.txt", 'w') as test:
        print("Generate test.txt...")
        for test_label in res_lines[-test_num:]:
            test_label = list(map(str, test_label))
            test.write(' '.join(test_label) + '\n')
        print("test.txt generated.")
    if check_txt:
        with open("../data/train.txt") as train:
            train_labels = train.readlines()
            print("Length of train.txt", len(train_labels))
            print(len(train_labels[0].split()), "elements in each line of train.txt.")
        with open("../data/valid.txt") as valid:
            valid_labels = valid.readlines()
            print("Length of valid.txt", len(valid_labels))
            print(len(valid_labels[0].split()), "elements in each line of valid.txt.")
        with open("../data/test.txt") as test:
            test_labels = test.readlines()
            print("Length of test.txt", len(test_labels))
            print(len(test_labels[0].split()), "elements in each line of test.txt.")


if __name__ == "__main__":
    # have a view of data
    random_view_data_and_label(img_folder_I, expand=True)
    random_view_data_and_label(img_folder_II, expand=True)
    random_view_data_and_label(img_folder_total, expand=True)

    # read all labels into 1 list
    all_label_in_list = list(map(str.split, load_metadata(data_folder, 'total_data')))
    # expand the bbox and alter the key points
    labels_altered = process_res_lines(all_label_in_list)
    # generate the train.txt, valid.txt, test.txt, they're split in 8:1:1
    output_new_labels(labels_altered, check_txt=True)
    # check the data again
    random_view_data_and_label('../data', cropped=True, txt_name='/train.txt')
    random_view_data_and_label('../data', cropped=True, txt_name='/valid.txt')
    random_view_data_and_label('../data', cropped=True, txt_name='/test.txt')