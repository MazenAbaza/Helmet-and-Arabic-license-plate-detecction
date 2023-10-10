# This is a sample Python script.

# Author: Krishnaragavan
# Date: 10/06/2021
# Import packages
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util



def detection(helmet_inference_path, frozen_graph_path, labelmap, number_of_classes, input):

    detection_graph = tf.Graph()
    TRAINED_MODEL_DIR = helmet_inference_path
    PATH_TO_CKPT = TRAINED_MODEL_DIR + frozen_graph_path
    PATH_TO_LABELS = TRAINED_MODEL_DIR + labelmap
    NUM_CLASSES = number_of_classes

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    print("> ====== Loading frozen graph into memory")

    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)
        print(">  ====== Inference graph loaded.")

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    img = input

    sess = tf.compat.v1.Session(graph=detection_graph)
    image = cv2.imread(img)
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print("Error: Image is empty or has zero dimensions.")
        return
    image = cv2.resize(image, (1080, 1080))
    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Excluded classes (modify this list as needed)
    included_classes = ['person', 'motorbike', 'with_helmet', 'without_helmet']

    # Filter out the excluded classes
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    for i in range(len(classes[0])):
        class_id = int(classes[0][i])
        if class_id in category_index and category_index[class_id]['name'] in included_classes:
            filtered_boxes.append(boxes[0][i])
            filtered_scores.append(scores[0][i])
            filtered_classes.append(class_id)

    filtered_category_index = {k: category_index[k] for k in filtered_classes}

    return filtered_category_index, image, filtered_boxes, filtered_scores, filtered_classes, num
