import torch
from os.path import join
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from math import cos, sin
from transforms import get_transform
import cv2
import math


if os.name == 'nt':
    root_dir = 'D:/data/pku-autonomous-driving'
else:
    root_dir = '/tmp/lly/pku-autonomous-driving'


class PUB_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', debug=False):
        self.root_dir = root_dir
        self.split = split
        self.transforms = {'train': get_transform(True),
                           'val': get_transform(False),
                           'test': get_transform(False),
                           }

        self.data = list()
        self.indices = None
        self.load_data()
        # 1686.2379、1354.9849为主点坐标（相对于成像平面）
        # 摄像机分辨率 3384*2710
        self.k = np.array([[2304.5479, 0, 1686.2379],
                           [0, 2305.8757, 1354.9849],
                           [0, 0, 1]], dtype=np.float32)
        self.debug = debug

    def load_data(self):
        data = pd.read_csv(join(self.root_dir, 'train.csv'))
        for (ImageId, PredictionString) in data.values:
            self.data.append({'ImageId': ImageId,
                              'PredictionString': PredictionString})

        sample_count = len(self.data)  # 训练集中样本的数量
        indices = np.arange(sample_count)
        np.random.seed(0)  # 固定随机种子
        np.random.shuffle(indices)
        if self.split == 'train':
            self.indices = indices[:sample_count // 5 * 4]
        else:
            self.indices = indices[-sample_count // 5:]

    def __getitem__(self, index):

        try:
            index = self.indices[index]
            sample_info = self.data[index]
            ImageId, PredictionString = sample_info['ImageId'], sample_info['PredictionString']
            items = PredictionString.split(' ')
            model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]

            rgb_path = join(self.root_dir, 'train_images', f'{ImageId}.jpg')
            mask_path = rgb_path.replace('images', 'masks')
            image = Image.open(rgb_path)
            flip_image = ImageOps.mirror(Image.open(rgb_path))
            has_mask = False
            try:
                mask = Image.open(mask_path)
                has_mask = True
            except Exception as e:
                pass
                # print(e)

            num_objs = len(model_types)
            boxes = []
            poses = []
            for model_type, yaw, pitch, roll, x, y, z in zip(model_types, yaws, pitches, rolls, xs, ys, zs):
                yaw, pitch, roll, xw, yw, zw = [float(x) for x in [yaw, pitch, roll, x, y, z]]
                # print(xw, yw, zw)
                yaw, pitch, roll = -pitch, -yaw, -roll  # 好像要变换一下
                poses.append([yaw/math.pi, pitch, roll/math.pi, zw/25])     # zw is too big
                img_cor_points = self.world_2_image(xw, yw, zw, yaw, pitch, roll)
                # zc就是zw
                # image_2_world(img_cor_points[0, :], self.k)    # 2d转3d

                if has_mask:
                    # center_x, center_y, _ = img_cor_points[0]
                    # print(center_x, center_y)
                    if not check_valid(mask, img_cor_points):
                        # print("太小了, 丢弃", box_area)
                        continue
                    # else:
                    #     print("合适的样本", box_area)

                xmin, ymin, xmax, ymax = cal_bbox(img_cor_points)
                boxes.append([xmin, ymin, xmax, ymax])

                if has_mask and self.debug:
                    image = draw_points(image, img_cor_points)
                    image = draw_line(image, img_cor_points)
                    image = draw_bbox(image, img_cor_points)

                    img_cor_points = self.world_2_image(-xw, yw, zw, -yaw, pitch, -roll)

                    flip_image = draw_points(flip_image, img_cor_points)
                    flip_image = draw_line(flip_image, img_cor_points)
                    flip_image = draw_bbox(flip_image, img_cor_points)

            if has_mask and self.debug:
                image_array = np.array(image)
                mask = np.array(mask)
                image_array[np.where((mask == [255, 255, 255]).all(axis=2))] = [255, 255, 255]
                image_array = cv2.resize(image_array, (1686, 1354))[:, :, ::-1]
                # mask = cv2.resize(mask, (1686, 1354))
                # cv2.imshow("mask", mask)

                # cv2.imshow("", image_array)
                # cv2.imshow('flip', cv2.resize(flip_image, (1686, 1354))[:, :, ::-1])
                # cv2.waitKey(1000)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            poses = torch.as_tensor(poses, dtype=torch.float32)

            labels = torch.ones((num_objs,), dtype=torch.int64)
            image_id = torch.tensor(index)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = dict()
            target["poses"] = poses
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                image, target = self.transforms[self.split](image, target)
            # 这里就要构造输入和输出了
            return image, target
        except Exception as e:
            print(e)

    def __len__(self):
        if os.name == 'nt':
            return 2
        return len(self.indices)

    def world_2_image(self, xw, yw, zw, yaw, pitch, roll):
        x_l = 1.02
        y_l = 0.80
        z_l = 2.31
        Rt = np.eye(4)
        t = np.array([xw, yw, zw])
        Rt[:3, 3] = t
        rot_mat = euler_to_R(yaw, pitch, roll).T
        #
        Rt[:3, :3] = rot_mat
        Rt = Rt[:3, :]
        rotation_vec, _ = cv2.Rodrigues(Rt[:3, :3])
        # print(yaw, pitch, roll, rotation_vec, zw/10)

        P = np.array([[0, 0, 0, 1],
                      [x_l, y_l, -z_l, 1],
                      [x_l, y_l, z_l, 1],
                      [-x_l, y_l, z_l, 1],
                      [-x_l, y_l, -z_l, 1],
                      [x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1]]).T
        img_cor_points = np.dot(self.k, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        return img_cor_points


def check_valid(mask, img_cor_points):

    try:
        for point in img_cor_points[:1]:
            center_x, center_y, _ = point
            r, g, b = mask.getpixel((int(center_x), int(center_y)))
            if (r, g, b) == (255, 255, 255):
                return False

    except Exception as e:
        # print(e)
        return False

    return True


def image_2_world(img_cor, K):
    x, y, z = img_cor
    xc, yc, zc = x*z, y*z, z
    p_cam = np.array([xc, yc, zc])
    T = np.dot(np.linalg.inv(K), p_cam)
    # print(T)
    # K_ = np.eye(4)
    # K_[:3, :3] = K
    # K_ = np.linalg.inv(K_)
    # Rt = np.dot(K_, P)
    # print(Rt)


def cal_bbox(points):
    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    return xmin, ymin, xmax, ymax


def draw_bbox(image, points):
    image = np.array(image)
    xmin, ymin, xmax, ymax = cal_bbox(points)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
    return image


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 2)
    cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, 2)

    cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, 2)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 2)
    cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, 2)
    cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, 2)
    cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, 2)

    cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, 2)
    cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, 2)

    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 2)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 2)
    cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, 2)
    return image


def draw_points(image, points):
    image = np.array(image)
    for (p_x, p_y, p_z) in points[:1]:
        # print("p_x, p_y", p_x, p_y)
        cv2.circle(image, (p_x, p_y), 15, (255, 0, 0), -1)
    return image


def euler_to_R(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    # return np.dot(R, np.dot(P, Y))
    return np.dot(Y, np.dot(P, R))


if __name__ == '__main__':
    max_z = 0
    dataset = PUB_Dataset(root_dir, 'train', debug=os.name == 'nt')
    for i in range(len(dataset)):
        image, target = dataset[i]
        # print(target['poses'])
        if target['poses'][:, 3].max() > max_z:
            max_z = target['poses'][:, 3].max()
        print(max_z)
        # print(target)
