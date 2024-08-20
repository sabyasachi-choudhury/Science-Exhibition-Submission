import os
import xml.etree.ElementTree as ET
import json
import cv2
import argparse

import keras
import tensorflow as tf
import numpy as np

ANNOTATIONS = False
IMAGE_PROCESS = True
JSON = False
TXT = True

"""NMS, postprocess, and iou check are for interpretation."""

if ANNOTATIONS:

    tree = ET.parse("apple_dataset/train/annotations/apple (1).xml")
    root = tree.getroot()
    print(tree)
    print(root)

    json_dict = {}

    print(os.listdir("apple_dataset/train/annotations"))

    if JSON:
        for i in range(1, 295):
            file = "apple (" + str(i) + ").xml"
            tree = ET.parse("apple_dataset/train/annotations/" + file)
            root = tree.getroot()
            d = {"size": [root[4][0].text, root[4][1].text, root[4][2].text]}
            names, top_left, down_right = [], [], []
            for item in root.findall("object"):
                names.append(item[0].text)
                top_left.append([int(item[4][0].text), int(item[4][1].text)])
                down_right.append([int(item[4][2].text), int(item[4][3].text)])
            d["name"] = names
            d["top_left"] = top_left
            d["down_right"] = down_right

            json_dict[file] = d
            print(file)

        for i in range(1, 270):
            file = "damaged_apple (" + str(i) + ").xml"
            tree = ET.parse("apple_dataset/train/annotations/" + file)
            root = tree.getroot()
            d = {"size": [root[4][0].text, root[4][1].text, root[4][2].text]}
            names, top_left, down_right = [], [], []
            for item in root.findall("object"):
                names.append(item[0].text)
                top_left.append([int(item[4][0].text), int(item[4][1].text)])
                down_right.append([int(item[4][2].text), int(item[4][3].text)])
            d["name"] = names
            d["top_left"] = top_left
            d["down_right"] = down_right

            json_dict[file] = d
            print(file)

        with open("train_ann.json", "w") as f:
            json.dump(json_dict, f, indent=4)

    # if TXT:
    #     with open("annotation.txt", "a") as f:
    #         for i in range(1, 295)


# Output shaoe:
# [(bsize, scale1, scale1, bounding_boxes_per_cell*(num_classes + 5)),
# (bsize, scale2, scale2, bounding_boxes_per_cell*(num_classes + 5))
# (bsize, scale3, scale3, bounding_boxes_per_cell*(num_classes + 5))]

def decode(out, i=0):
    out = out[i]
    batch_size = out.shape[0]
    grid_size = out.shape[1]
    num_classes = 80
    num_anchors = 3

    out = tf.reshape(out, [batch_size, grid_size, grid_size, num_anchors, -1])

    y = tf.range(grid_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, grid_size])
    x = tf.range(grid_size, dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [grid_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, num_anchors, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    out_center = out[..., :2]  # center coords
    out_dims = out[..., 2:4]  # dimensions of box
    out_conf = out[..., 4:5]  # confidence of object being present
    out_probs = out[..., 5:]  # probability for each class
    print(out_center.shape)
    print(out_conf.shape)

    pred_xy = (tf.sigmoid(out_center) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(out_dims) * ANCHORS[i]) * STRIDES[i] * 8
    pred_conf = tf.sigmoid(out_conf)
    pred_probs = tf.sigmoid(out_probs)
    print(pred_xy.shape, pred_wh.shape, pred_conf.shape, pred_probs.shape)

    return tf.concat([pred_xy, pred_wh, pred_conf, pred_probs], axis=-1)

# pred_bbox is list of boxes predicted by decode with obj_score > thresh limit
def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    # resize_ratio = min(input_size / org_w, input_size / org_h)
    x_resize_ratio, y_resize_ratio = input_size/org_w, input_size/org_h
    print(x_resize_ratio, y_resize_ratio)

    pred_coor[:, 0:3:2] = pred_coor[:, 0:3:2] / x_resize_ratio
    pred_coor[:, 1:4:2] = pred_coor[:, 1:4:2] / y_resize_ratio
    print(pred_coor)

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    print(pred_coor)

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold):
    # param bboxes: (xmin, ymin, xmax, ymax, score, class), converted by box_postprocess
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:  # nms is applied class-wise
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            # cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)
            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


IMG_SIZE = 416
ANCHORS = [[[10, 13], [16, 30], [33, 23]],
           [[30, 61], [62, 45], [59, 119]],
           [[116, 90], [156, 198], [373, 326]]]
STRIDES = np.array([32, 16, 8])
ANCHORS = (np.array(ANCHORS).T / STRIDES).T
with open("coco.names", "r") as file:
    CLASS_NAMES = file.read().split(sep="\n")
images = [cv2.imread("dog.jpg"), cv2.imread("eagle.jpg")]
print(images[0].shape)
for i, img in enumerate(images):
    images[i] = cv2.resize(img, [IMG_SIZE, IMG_SIZE], cv2.INTER_AREA) / 255

model = keras.models.load_model("pretrained-yolov3.h5")
prediction = model.predict(np.array(images))
print(prediction[0].shape)
print(prediction[1].shape)
print(prediction[2].shape)
p = decode(prediction, i=0)

thresh_boxes = []
for row in range(13):
    for col in range(13):
        cell = p[0][row][col]
        for anc in cell:
            if anc[4] >= 0.5:
                thresh_boxes.append(anc)

im = cv2.imread("dog.jpg")
bboxes = postprocess_boxes(thresh_boxes, im, IMG_SIZE, 0.5)
bboxes = nms(bboxes, 0.6)
print(bboxes)
for box in bboxes:
    im = cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
cv2.imshow("sd", im)
cv2.waitKey(0)