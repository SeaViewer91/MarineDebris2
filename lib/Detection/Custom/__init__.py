import os
import re
import numpy as np
import json

import cv2

import tensorflow as tf

from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model, Input
from keras.callbacks import TensorBoard
import keras.backend as K

# Detection을 위한 라이브러리(lib)
# Reference 
# https://github.com/experiencor/keras-yolo3
# https://github.com/qqwweee/keras-yolo3
# https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
# https://github.com/experiencor/keras-yolo3/blob/master/yolo3_one_file_to_detect_them_all.py
from lib.Detection.Custom.voc import parse_voc_annotation
from lib.Detection.Custom.yolo import create_yolov3_model, dummy_loss
from lib.Detection.YOLOv3.models import yolo_main
from lib.Detection.Custom.generator import BatchGenerator
from lib.Detection.Custom.utils.utils import normalize, evaluate, makedirs
from lib.Detection.Custom.callbacks import CustomModelCheckpoint, CustomTensorBoard
from lib.Detection.Custom.utils.multi_gpu_model import multi_gpu_model
from lib.Detection.Custom.gen_anchors import generateAnchors


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ObjectDetection:

    def __init__(self):
        self.__model_type = ""
        self.__model_path = ""
        self.__model_labels = []
        self.__model_anchors = []
        self.__detection_config_json_path = ""
        self.__input_size = 416 # input size는 32의 배수 중 다른 수로 변경하여 사용 가능 
        self.__object_threshold = 0.4
        self.__nms_threshold = 0.4
        self.__model = None
        self.__detection_utils = CustomDetectionUtils(labels=[])

    def YOLOv3(self):

        self.__model_type = "yolov3"

    def PretrainedModelPath(self, detection_model_path):

        self.__model_path = detection_model_path

    def MetadataPath(self, configuration_json):

        self.__detection_config_json_path = configuration_json

    def loadModel(self):

        if self.__model_type == "yolov3":
            detection_model_json = json.load(open(self.__detection_config_json_path))

            self.__model_labels = detection_model_json["labels"]
            self.__model_anchors = detection_model_json["anchors"]

            self.__detection_utils = CustomDetectionUtils(labels=self.__model_labels)

            self.__model = yolo_main(Input(shape=(None, None, 3)), 3, len(self.__model_labels))

            self.__model.load_weights(self.__model_path)

    def ImageDetector(
                    self, 
                    input_image="", 
                    output_image_path="", 
                    input_type="file", 
                    output_type="file",
                    extract_detected_objects=False, 
                    minimum_confidence=50, 
                    nms_treshold=0.4,
                    display_confidence=True, 
                    display_object_name=True, 
                    thread_safe=False):

        if self.__model is None:
            raise ValueError("Call .loadModel() function")
        else:
            if output_type == "file":
                output_image_folder, n_subs = re.subn(r'\.(?:jpe?g|png|tif)$', '', output_image_path, flags=re.I)
                if n_subs == 0:
                    raise ValueError("output_image_path error. "
                                     "Therefore it must end as one the following: "
                                     "'.jpeg', '.jpg', '.png', '.tif'. {} found".format(output_image_path))
                elif extract_detected_objects:
                    objects_dir = output_image_folder + "-objects"
                    os.makedirs(objects_dir, exist_ok=True)

            self.__object_threshold = minimum_confidence / 100
            self.__nms_threshold = nms_treshold

            output_objects_array = []
            detected_objects_image_array = []

            if input_type == "file":
                image = cv2.imread(input_image)
            elif input_type == "array":
                image = input_image
            else:
                raise ValueError("input_type error : input image must be 'file(.jpg, .png, .tif)' or 'array(Tensor)'. {} found".format(input_type))

            image_frame = image.copy()

            height, width, channels = image.shape

            image = cv2.resize(image, (self.__input_size, self.__input_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float32") / 255.

            # expand the image to batch
            image = np.expand_dims(image, 0)

            if self.__model_type == "yolov3":
                if thread_safe == True:
                    with K.get_session().graph.as_default():
                        yolo_results = self.__model.predict(image)
                else:
                    yolo_results = self.__model.predict(image)

                boxes = list()

                for idx, result in enumerate(yolo_results):
                    box_set = self.__detection_utils.decode_netout(result[0], self.__model_anchors[idx],
                                                                   self.__object_threshold, self.__input_size,
                                                                   self.__input_size)
                    boxes += box_set

                self.__detection_utils.correct_yolo_boxes(boxes, height, width, self.__input_size, self.__input_size)
                self.__detection_utils.do_nms(boxes, self.__nms_threshold)

                all_boxes, all_labels, all_scores = self.__detection_utils.get_boxes(boxes, self.__model_labels,
                                                                                     self.__object_threshold)

                for object_box, object_label, object_score in zip(all_boxes, all_labels, all_scores):
                    each_object_details = dict() # output format : dictionary
                    each_object_details["name"] = object_label
                    each_object_details["confidence"] = object_score

                    if object_box.xmin < 0:
                        object_box.xmin = 0
                    if object_box.ymin < 0:
                        object_box.ymin = 0

                    each_object_details["boxinfo"] = [object_box.xmin, object_box.ymin, object_box.xmax, object_box.ymax]
                    output_objects_array.append(each_object_details)

                drawn_image = self.__detection_utils.draw_boxes_and_caption(image_frame.copy(), all_boxes, all_labels,
                                                                            all_scores, show_names=display_object_name,
                                                                            show_percentage=display_confidence)

                if extract_detected_objects:

                    for cnt, each_object in enumerate(output_objects_array):

                        splitted_image = image_frame[each_object["boxinfo"][1]:each_object["boxinfo"][3],
                                                     each_object["boxinfo"][0]:each_object["boxinfo"][2]]
                        if output_type == "file":
                            splitted_image_path = os.path.join(objects_dir, "{}-{:05d}.jpg".format(each_object["name"],
                                                                                                   cnt))

                            cv2.imwrite(splitted_image_path, splitted_image)
                            detected_objects_image_array.append(splitted_image_path)
                        elif output_type == "array":
                            detected_objects_image_array.append(splitted_image.copy())

                if output_type == "file":

                    cv2.imwrite(output_image_path, drawn_image)

                if extract_detected_objects:
                    if output_type == "file" or output_type == "info":
                        return output_objects_array, detected_objects_image_array
                    elif output_type == "array":
                        return drawn_image, output_objects_array, detected_objects_image_array

                else:
                    if output_type == "file" or output_type == "info":
                        return output_objects_array
                    elif output_type == "array":
                        return drawn_image, output_objects_array


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class CustomDetectionUtils:
    def __init__(self, labels):
        self.__labels = labels
        self.__colors = []

        for i in range(len(labels)):
            color_space_values = np.random.randint(50, 255, size=(3,))
            red, green, blue = color_space_values
            red, green, blue = int(red), int(green), int(blue)
            self.__colors.append([red, green, blue])

    @staticmethod
    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def decode_netout(self, netout, anchors, obj_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = self._sigmoid(netout[..., :2])
        netout[..., 4:] = self._sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):

                    objectness = netout[row, col, b, 4]

                    if objectness <= obj_thresh:
                        continue

                    x, y, w, h = netout[row, col, b, :4]
                    x = (col + x) / grid_w  # center position, unit: image width
                    y = (row + y) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

                    classes = netout[row, col, b, 5:]
                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                    boxes.append(box)

        return boxes

    @staticmethod
    def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
        new_w, new_h = net_w, net_h
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
            y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union = w1 * h1 + w2 * h2 - intersect

        try:
            result = float(intersect) / float(union)
            return result
        except:
            return 0.0

    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return

        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i].classes[c] == 0: continue

                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0

    def get_boxes(self, boxes, labels, thresh):
        v_boxes, v_labels, v_scores = list(), list(), list()

        for box in boxes:

            for i in range(len(labels)):

                if box.classes[i] > thresh:
                    v_boxes.append(box)
                    v_labels.append(labels[i])
                    v_scores.append(box.classes[i] * 100)

        return v_boxes, v_labels, v_scores

    def label_color(self, label):
        
        if label < len(self.__colors):
            return self.__colors[label]
        else:
            return 0, 255, 0

    def draw_boxes_and_caption(self, image_frame, v_boxes, v_labels, v_scores, show_names=False, show_percentage=False):

        for i in range(len(v_boxes)):
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            width, height = x2 - x1, y2 - y1
            class_color = self.label_color(self.__labels.index(v_labels[i]))

            image_frame = cv2.rectangle(image_frame, (x1, y1), (x2, y2), class_color, 2)

            label = ""
            if show_names and show_percentage:
                label = "%s : %.3f" % (v_labels[i], v_scores[i])
            elif show_names:
                label = "%s" % (v_labels[i])
            elif show_percentage:
                label = "%.3f" % (v_scores[i])

            if show_names or show_percentage:
                b = np.array([x1, y1, x2, y2]).astype(int)
                cv2.putText(image_frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 0), 3)
                cv2.putText(image_frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        return image_frame
