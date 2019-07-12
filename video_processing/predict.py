from pyspark import SparkContext
import argparse
import tensorflow as tf
import tarfile
import os
import numpy as np
import cv2


def predict(index, image):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('{}/{}/frozen_inference_graph.pb'.format(path, model_name), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as session:
            image_np_expanded = np.expand_dims(image, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            resp = {
                "index": index,
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "num_detections": num_detections
            }
            print(resp)


def extract_images(video_loc, index):
    vidcap = cv2.VideoCapture(video_loc)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, image = vidcap.read()
    if success:
        return index, image
    else:
        raise Exception('invalid frame')


def get_frame_count(path):
    v = cv2.VideoCapture(path)
    return int(v.get(cv2.CAP_PROP_FRAME_COUNT))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process video to create images')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('model_name', type=str, help='Model name to load (exclude .tar.gz)')
    # parser.add_argument('model_path', type=str, help='Path to model', default='.', required=False)

    args = parser.parse_args()

    video_path = args.video
    path = '.'
    model_name = args.model_name

    frame_count = get_frame_count(video_path)

    tar_file = tarfile.open('{}/{}.tar.gz'.format(path, model_name))
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
            
    sc = SparkContext('local', 'video_process_predict')
    count = sc.parallelize(range(0, frame_count)).map(lambda x: extract_images(video_path, x))\
        .filter(lambda x: predict(*x)).count()

