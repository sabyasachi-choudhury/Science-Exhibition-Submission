import json
import pickle

import cv2
import tensorflow as tf
import numpy as np

ANCHORS = [[[10, 13], [16, 30], [33, 23]],
           [[30, 61], [62, 45], [59, 119]],
           [[116, 90], [156, 198], [373, 326]]]

STRIDES = [32, 16, 8]
INPUT_SIZE = 416
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
ANCHOR_PER_SCALE = 3
MAX_BBOX_PER_SCALE = 100
IOU_LOSS_THRESH = 0.45
SCORE_THRESHOLD = 0.40
TRAIN_WARMUP_EPOCHS = 2
TRAIN_EPOCHS = 50
TRAIN_LR_INIT = 1e-4
TRAIN_LR_END = 1e-6

CLASSES = ["apple", "damaged_apple"]


class TrainBatchGenerator:
    def __init__(self):
        self.input_size = INPUT_SIZE
        self.batch_size = TRAIN_BATCH_SIZE
        self.strides = np.array(STRIDES)
        self.classes = CLASSES
        self.num_classes = len(self.classes)
        self.anchors = (np.array(ANCHORS).T / self.strides).T
        self.anchor_per_scale = ANCHOR_PER_SCALE
        self.max_bbox_per_scale = MAX_BBOX_PER_SCALE

        with open("train_data_store.pkl", "rb") as file:
            self.x_train, self.y_train = pickle.load(file)
        self.x_train = self.x_train.astype(np.float32)
        self.x_train /= 255.0
        for k, img_boxes in enumerate(self.y_train):
            for l, box in enumerate(img_boxes):
                if box[-1] == 'apple':
                    self.y_train[k][l][-1] = 0
                else:
                    self.y_train[k][l][-1] = 1
        self.num_samples = len(self.x_train)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        for i, x in enumerate(self.y_train):
            self.y_train[i] = np.array(x, dtype=np.float32)

        self.batch_count = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        with tf.device('/cpu:0'):
            self.output_size = self.input_size // self.strides
            batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3), dtype=np.float32)
            batch_label_sbbox = np.zeros((self.batch_size, self.output_size[0], self.output_size[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.output_size[1], self.output_size[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.output_size[2], self.output_size[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    image, bboxes = self.x_train[index], self.y_train[index]
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes
                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                k = np.random.permutation(len(self.x_train))
                self.x_train, self.y_train = self.x_train[k], [self.y_train[n] for n in k]
                raise StopIteration

    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.output_size[i], self.output_size[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))
        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            onehot = np.zeros(self.num_classes, dtype=np.float32)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def bbox_iou(self, boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        return inter_area * 1.0 / tf.maximum(union_area, 1e-24)


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # Calculate the iou value between the two bounding boxes
    iou = inter_area / tf.maximum(union_area, 1e-24)
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]
    # Calculate the GIoU value according to the GioU formula
    giou = iou - 1.0 * (enclose_area - union_area) / tf.maximum(enclose_area, 1e-24)
    return giou


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return 1.0 * inter_area / tf.maximum(union_area, 1e-24)


def compute_loss(pred, conv, label, bboxes, i=0, classes=CLASSES):
    NUM_CLASS = len(classes)
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Find the value of IoU with the real box The largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)  # .5

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss,


def create_data():
    with open("train_ann.json", "r") as file:
        json_data = json.load(file)
    x_data, y_data = [cv2.resize(cv2.imread("apple_dataset/train/images/" + x[:-3] + "jpg"), [INPUT_SIZE, INPUT_SIZE],
                                 interpolation=cv2.INTER_AREA) for x in json_data.keys()], []
    for i, key in enumerate(json_data.keys()):
        y_sub = []
        x_resize_ratio, y_resize_ratio = INPUT_SIZE / x_data[i].shape[0], INPUT_SIZE / x_data[i].shape[1]
        for j in range(len(json_data[key]["name"])):
            y_sub.append([
                json_data[key]["top_left"][j][0] * x_resize_ratio,
                json_data[key]["top_left"][j][1] * y_resize_ratio,
                json_data[key]["down_right"][j][0] * x_resize_ratio,
                json_data[key]["down_right"][j][1] * y_resize_ratio,
                json_data[key]["name"][j]
            ])
        y_data.append(y_sub)
    x_data, y_data = np.array(x_data), y_data
    with open("train_data_store.pkl", "wb") as file:
        pickle.dump([x_data, y_data], file)

def freeze(mod):
    for i in range(105):
        try:
            mod.get_layer("conv_" + str(i)).trainable=False
        except ValueError:
            pass
        try:
            mod.get_layer("bnorm_" + str(i)).trainable = False
        except ValueError:
            pass


trainset = TrainBatchGenerator() #training batch generator
model = tf.keras.models.load_model("pretrained-yolov3.h5")
freeze(model) # finetuning
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6) #optimizer to use

train_lr_init=TRAIN_LR_INIT
train_lr_end=TRAIN_LR_END
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
total_steps = TRAIN_EPOCHS * steps_per_epoch


def train_step(epoch, image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]
        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))
        global_steps.assign_add(1)
        # learning rate will be changed as a cosine function.
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * TRAIN_LR_INIT
        else:
            lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
        optimizer.lr.assign(lr.numpy())

for epoch in range(TRAIN_EPOCHS):
    for image_data, target in trainset:
        train_step(epoch, image_data, target)