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
def random_view_data_and_label(img_folder, expand=False, ratio=4, cropped=False, key_pts=True, txt_name='/label.txt'):
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
    # image_path = os.path.join(img_folder, split_line[0])
    image_path = split_line[0]
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
    if key_pts:
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
    bbox: a list, [x1, y1, x2, y2]
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
    lines: all the content in the txt, [sentence 1, setence 2, ....]

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

    int the txt of labels, every line is a string like below:
    "../data/xxxx/xxx.jpg x1 y1 x2 y2 kp1_x kp1_y.....kp21_x kp21_y"
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


def add_cls_label(origin_txt, target_txt, prefix="", suffix=' 1\n'):
    """
    :param origin_txt: the base txt to add label
    :param target_txt: the target txt to write
    :param prefix: the label to add on the head
    :param suffix: the label to add on the tail
    :return: nothing, just write something on the target_txt
    """
    with open(origin_txt, 'r+') as ori_txt:
        lines = ori_txt.readlines()
    print("Copying", origin_txt)
    with open(target_txt, 'a') as tgt_txt:
        cnt = 1
        for line in lines:
            print("line", cnt, line)
            tgt_txt.write(prefix + line.strip() + suffix)
            cnt += 1


def generate_negative_example(label_in_list, tgt_txt, neg_example_size=2288):
    """
    random crop in a img and output its corner coordinate into a txt

    :param label_in_list:
    a list of labels,[[file_path, x1, y1, x2, y2, keypoint1_x, keypoint1_y, ..., keypoint21_x, keypoint21_y],..]
    :param tgt_txt: the txt to record the information random crop
    :param tgt_txt: how many valid random crop results there should be
    :return: nothing, just write information to txt
    """
    # a dict {'xxx.jpg': [(x1, y1, x2, y2, H, W)...], ...}
    jpg_recs = {}

    # write the information of label rectangle into dict
    for res_line in label_in_list:
        print("res_line[0]", res_line[0])
        img = cv2.imread(res_line[0])
        res_line[1:5] = expand_roi(list(map(float, res_line[1:5])), img.shape)
        shape_list = img.shape[0: 2] # H * W * C
        del img
        if res_line[0] not in jpg_recs:
            jpg_recs[res_line[0]] = []
        print("shape_list", shape_list)
        print("res_line[1: 5]", res_line[1: 5])
        jpg_recs[res_line[0]].append(tuple(res_line[1: 5] + list(shape_list)))
        print("jpg_recs[res_line[0]]", jpg_recs[res_line[0]])
    # crop and check the IoU of the result
    # img_name xxx.jpg, recs [(x1, y1, x2, y2, H, W)], ...
    crop_cnt = 0
    while crop_cnt < neg_example_size:
        for img_name, recs in jpg_recs.items():
            print("img_name, recs: ", img_name, recs)
            # random select a point within the range of the shape
            random_r = random.randint(int(0.3 * recs[0][4]), int(0.7 * recs[0][4]))
            random_c = random.randint(int(0.3 * recs[0][5]), int(0.7 * recs[0][5]))
            size_h = random.randint(int(0.2 * recs[0][4]), int(0.3 * recs[0][4]))
            size_w = random.randint(int(0.2 * recs[0][5]), int(0.3 * recs[0][5]))

            # if IoU is OK, write name of picture and rec to the txt
            rand_rec = (random_c, random_r, random_c + size_w, random_r + size_h)
            iou_pass = True
            for rec in recs:
                if cal_iou(rec[0: 4], rand_rec) >= 0.15:
                    iou_pass = False
                    break
            print('IoU_pass:', iou_pass)
            if iou_pass:
                crop_cnt += 1
                with open(tgt_txt, "a") as tgt:
                    tgt.write(img_name + " " + " ".join(map(str, list(rand_rec))) + "\n")
            print('Crop_cnt:', crop_cnt)
            if crop_cnt >= neg_example_size:
                break


def cal_iou(rec1, rec2, eps=0.00001):
    """
    Calculate the IoU of 2 rectangles.

    :param rec1: a tuple representing rec1 (x1, y1, x2, y2)
    :param rec2: a tuple representing rec2 (x3, y3, x4, y4)
    :param eps: avoid divided by zero
    :return: the IoU between rec1 and rec2
    """
    if len(rec1) != 4 or len(rec2) != 4:
        print("Invalid Input !!")
        return
    x1, y1, x2, y2 = rec1
    x3, y3, x4, y4 = rec2
    # don't forget + 1
    area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area2 = (x4 - x3 + 1) * (y4 - y3 + 1)
    # don't forget take the relative position into account
    inter_w = max(0, min(x4, x2) - max(x1, x3) + 1)
    inter_h = max(0, min(y4, y2) - max(y1, y3) + 1)
    intersection = inter_h * inter_w
    union = area1 + area2 - intersection
    return intersection / (union + eps)  # avoid divide zero


def merge_pos_neg(pos_txt, neg_txt, final_txt):
    final_t = []
    with open(pos_txt, 'r') as pos_t:
        final_t.extend(pos_t.readlines())
    with open(neg_txt, 'r') as neg_t:
        final_t.extend(neg_t.readlines())
    random.shuffle(final_t)

    for label in final_t:
        with open(final_txt, 'a+') as final_t:
            final_t.write(label)


if __name__ == "__main__":
    """
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
    
    # add the classification label to the positive example
    add_cls_label('../data/train.txt', '../data/train_cls.txt')
    add_cls_label('../data/valid.txt', '../data/valid_cls.txt')
    add_cls_label('../data/test.txt', '../data/test_cls.txt')
    
    # random crop to generate negative examples
    generate_negative_example(all_label_in_list,
                              '../data/negative_examples/negative_examples.txt')
    
    # check if the negative examples is OK                          
    random_view_data_and_label(img_folder_total, expand=False, key_pts=False, txt_name='/negative_examples.txt')
    
    # add the classification label to the negative examples
    add_cls_label('../data/negative_examples.txt', '../data/negative_examples_cls.txt', suffix=' 0\n')
    
    # i did the split 8 : 1: 1 of negative example manually haha
    # merge the positive example and negative example together
    merge_pos_neg("../data/train_cls_pos.txt",
                  "../data/train_cls_neg.txt",
                  "../data/train_cls_total.txt")
    merge_pos_neg("../data/valid_cls_pos.txt",
                  "../data/valid_cls_neg.txt",
                  "../data/valid_cls_total.txt")
    merge_pos_neg("../data/test_cls_pos.txt",
                  "../data/test_cls_neg.txt",
                  "../data/test_cls_total.txt")
    # check the result of merge
    for _ in range(30):
        random_view_data_and_label(img_folder_total, expand=False, key_pts=False, txt_name='/train_cls_total.txt')
    for _ in range(30):
        random_view_data_and_label(img_folder_total, expand=False, key_pts=False, txt_name='/valid_cls_total.txt')
    for _ in range(30):
        random_view_data_and_label(img_folder_total, expand=False, key_pts=False, txt_name='/test_cls_total.txt')
    """
    print("Finish!")
