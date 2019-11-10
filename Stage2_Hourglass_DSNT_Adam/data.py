import numpy as np
import cv2
import math
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools
import random

folder_list = ['I', 'II']
train_boarder = 128


def channel_norm(img):
    """
    Normalize an image.

    :param img: a single channel image matrix.
    :return: a normalized image matrix
    """
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    """
    Parse information from a line in label.txt.
    :param line: a line in txt is in the format like [image_path, rectangle corner (x, y)* 2, keypoint (x, y) * 21]
    :return: return the img_name, rectangle corner point, landmarks seperately.
    """
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 128 x 128
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        """
        :param sample: raw images cropped according to the rectangle and landmarks, dict[key: nparray]
        :return: the resized and normalized images and landmarks, dict[key: nparray]
        """
        # image and landmarks here are all represent in nparray
        image, landmarks = sample['image'], sample['landmarks']
        # print("Normalize input size:", image)
        image_resize = np.asarray(
                            image.resize((train_boarder, train_boarder), Image.BILINEAR),
                            dtype=np.float32)       # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        # print("Normalize output size:", image.shape)
        # the values in the dict are still np arrays
        return {'image': image,
                'landmarks': landmarks
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        """
        :param sample: the images in same size and landmarks, dict[key: nparray]
        :return: images and landmarks, dict[key: tensor]
        """
        # values of the dict are all np arrays
        image, landmarks = sample['image'], sample['landmarks']

        # numpy image: H x W x C, torch image: C X H X W
        # pay attention do the operator image = image.transpose((2, 0, 1)), if necessary.
        # print("ToTensor input size:", image.shape)

        # input is in the size (train_boarder, train_boarder), so translate it to (1, train_boarder, train_boarder)
        image = np.expand_dims(image, axis=0)
        # print("ToTensor output size:", image.shape)

        # values of dict are now all tensors
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

# some bug here
class RandomVerticalFlip(object):
    """
        Do the Vertical flip with a possibility.
    """
    def __call__(self, sample, p=0.5):
        """
        :param sample: images and landmarks, dict[key: tensor]
        :param p: the possibility to the vertical flip
        :return: the transformed images and corresponding landmarks in dict [key: tensor]
        """
        image, landmarks = sample['image'], sample['landmarks']
        # print("HorizonFlip input size:", image.shape)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))
        if random.random() < p:
            image = torch.from_numpy(image.numpy()[:, ::-1, :].copy())
            landmarks = landmarks.reshape(-1, 2)
            landmarks[:, 1] = train_boarder - landmarks[:, 1]
        # print("HorizonFlip output size:", image.shape)
        return {'image': image,
                'landmarks': landmarks.flatten()}


class RandomHorizonalFlip(object):
    """
        Do the Horizonal flip with a possibility.
    """
    def __call__(self, sample, p=0.5):
        """
        :param sample: images and landmarks, dict[key: tensor]
        :param p: the possibility to the horizonal flip
        :return: the transformed images and corresponding landmarks in dict [key: tensor]
        """
        image, landmarks = sample['image'], sample['landmarks']
        # print("HorizonFlip input size:", image.shape)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))

        if random.random() < p:
            image = torch.from_numpy(image.numpy()[:, :, ::-1].copy())
            landmarks = landmarks.reshape(-1, 2)
            # print("origin landmarks: ", landmarks)
            landmarks[:, 0] = train_boarder - landmarks[:, 0]
            # just tranlate landmarks coordinates is not enough, we must reorginized the order of coordinate
            landmarks_ = torch.zeros((landmarks.shape[0], landmarks.shape[1]))
            # point 1-6 elbow
            landmarks_[0, :] = landmarks[5, :]
            landmarks_[1, :] = landmarks[4, :]
            landmarks_[2, :] = landmarks[3, :]
            landmarks_[3, :] = landmarks[2, :]
            landmarks_[4, :] = landmarks[1, :]
            landmarks_[5, :] = landmarks[0, :]
            # point 7 - 10 eye out corner
            landmarks_[6, :] = landmarks[9, :]
            landmarks_[7, :] = landmarks[8, :]
            landmarks_[8, :] = landmarks[7, :]
            landmarks_[9, :] = landmarks[6, :]
            # point 11 13 nose corner
            landmarks_[10, :] = landmarks[12, :]
            landmarks_[12, :] = landmarks[10, :]
            # point 17 18 eye center
            landmarks_[16, :] = landmarks[17, :]
            landmarks_[17, :] = landmarks[16, :]
            # point 20 21 mouth corner
            landmarks_[19, :] = landmarks[20, :]
            landmarks_[20, :] = landmarks[19, :]
            # other points
            landmarks_[11, :] = landmarks[11, :]
            landmarks_[13, :] = landmarks[13, :]
            landmarks_[14, :] = landmarks[14, :]
            landmarks_[15, :] = landmarks[15, :]
            landmarks_[18, :] = landmarks[18, :]

            # print("Flip landmarks: ", landmarks_)
            landmarks = landmarks_.flatten()
        # print("HorizonFlip output size:", image.shape)
        return {'image': image,
                'landmarks': landmarks}


class SimilarityTransform(object):
    """
    Do the similarity transform,namely, a random rotation and a random translation.
    """

    def __call__(self, sample, angle=30.0, trans_per=0.1):
        """
        :param sample: images and landmarks, dict[key: tensor]
        :param angle: random rotation angle range. e.g. 30 means rotate random angle from [-30, 30]
        :param trans_per: relative percentage of transform to the border
        :return: the transformed images and corresponding landmarks in dict [key: tensor]
        """
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))
        rot_deg = (random.random() * 2 - 1) * angle
        rot_rad = rot_deg / 180.0 * math.pi
        trans_ofst_x = (random.random() * 2 - 1) * trans_per * image.shape[2]
        trans_ofst_y = (random.random() * 2 - 1) * trans_per * image.shape[1]

        img = image.numpy().transpose((1, 2, 0))
        # rotate
        rot_M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rot_deg, 1)
        img_rotate = cv2.warpAffine(img, rot_M, (img.shape[1], img.shape[0]))
        landmarks = landmarks.numpy().reshape(-1, 2).T  # (21, 2)
        landmarks = np.vstack([landmarks, np.ones(landmarks.shape[1])])

        landmarks = ((np.vstack([rot_M, [0, 0, 1]]) @ landmarks)[0:2, :]).T  # (21, 2)
        # translate
        img_trans = np.array([[1, 0, trans_ofst_x], [0, 1, trans_ofst_y]])
        img_trans = cv2.warpAffine(img_rotate, img_trans, (img_rotate.shape[1], img_rotate.shape[0]))
        landmarks[:, 0] += trans_ofst_x
        landmarks[:, 1] += trans_ofst_y
        landmarks = landmarks.ravel().astype(np.float32)
        img_trans = np.expand_dims(img_trans, axis=0)

        return {'image': torch.from_numpy(img_trans),
                'landmarks': torch.from_numpy(landmarks)}


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, phase, transform=None):
        """
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        """
        self.lines = src_lines
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx, train_boarder=128, ep=10e-8):
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        img = Image.open(img_name).convert('L') # notice, here is a gray image

        img_crop = img.crop(tuple(rect))            
        landmarks = np.array(landmarks).astype(np.float32)

        # let landmarks fit to the train_boarder(112)
        landmarks = landmarks.reshape([-1, 2])
        landmarks[:, 0] = landmarks[:, 0] * train_boarder / (rect[2] - rect[0] + ep)
        landmarks[:, 1] = landmarks[:, 1] * train_boarder / (rect[3] - rect[1] + ep)
        landmarks = landmarks.flatten()

        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)
        return sample


def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor(),
            # RandomVerticalFlip(),
            RandomHorizonalFlip(),
            SimilarityTransform()
        ]
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()]
        )
    data_set = FaceLandmarksDataset(lines, phase, transform=tsfm)
    return data_set


def get_train_valid_set():
    train_data = load_data('train')
    valid_data = load_data('valid')
    test_data = load_data('test')
    return train_data, valid_data, test_data


if __name__ == '__main__':
    train_set = load_data('train')
    for i in range(0, len(train_set)):
        sample = train_set[i]
        img = sample['image']
        landmarks = sample['landmarks']
        # print(img.shape)
        img = img.numpy().transpose((1, 2, 0))

        # change from gray to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for j in range(0, len(landmarks), 2):
            # print((landmarks[j], landmarks[j + 1]))
            cv2.circle(img, (landmarks[j], landmarks[j + 1]), 2, (0, 0, 255), -1)
        # print(landmarks)
        cv2.imshow('Check the keypoint and image' + str(i), img)

        key = cv2.waitKey()
        if key == 27:
            exit(0)
        cv2.destroyAllWindows()






