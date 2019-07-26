from pyspark import SparkContext
import argparse
import tensorflow as tf
import tarfile
import os
import shutil
import numpy as np
import cv2
import uuid


def denormalize_bbox(bbox_floats, image_size):
    """Converts bbox with normalized 0-1 coordinates to [x,y,w,h] format.
    Args:
        bbox_floats (np.array): array of 4 floats from 0-1 that give coordinates
                                for [y1, x1, y2, x2] normalized for image size.
        image_size (list): Original size of image as [width, height] so the
                           absolute bbox coordinates can be computed.
    Returns:
        list(int): list of bounding box absolute coordinates in [x1, y1, x2, y2] format
    """
    bbox_ints = (np.fliplr(bbox_floats.reshape(2, 2)) * image_size[:2][::-1]).astype(int)
    bbox = [int(x) for x in bbox_ints.reshape(-1)]

    return bbox


def predict(index, image):
    with tf.Graph().as_default() as g:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_data_bc.value)
        tf.import_graph_def(graph_def, name='')

        with tf.Session(graph=g) as session:
            image_np_expanded = np.expand_dims(image, axis=0)
            image_tensor = session.graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = session.graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = session.graph.get_tensor_by_name('detection_scores:0')
            classes = session.graph.get_tensor_by_name('detection_classes:0')
            num_detections = session.graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            return {
                "index": index,
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "num_detections": num_detections
            }, image.shape


def process_result(result, image_shape):
    num_detections = int(result['num_detections'][0])

    scores = result['scores'][0][:num_detections]
    object_ids = list(range(1, num_detections + 1))

    # bbboxes are normalized [0, 1), ymin, xmin, ymax, xmax
    bboxes = result['boxes'][0][:num_detections]
    denormalize_bbox_arr = []
    for coord_set in bboxes:
        denormalize_bbox_arr.append(denormalize_bbox(coord_set, image_shape))

    # Convert from (x1,y1,x2,y2) to (x,y,w,h)
    denormalized_bboxes = [[x1, y1, x2 - x1, y2 - y1] for
                           (x1, y1, x2, y2) in denormalize_bbox_arr]

    # convert class indexes into string (ie 'person' or 'car')
    det_classes = result['classes'][0][:num_detections]
    classnames = [str(int(d)) for d in det_classes]

    # create detection dictionary
    output_dict = {
        'scores': scores,
        'classnames': classnames,
        'bboxes': denormalized_bboxes,
        'client_object_ids': object_ids,
    }

    detection_list = []
    if output_dict:
        for classname, bbox, score, obj_id in zip(output_dict.get('classnames'),
                                                  output_dict.get('bboxes'),
                                                  output_dict.get('scores'),
                                                  output_dict.get('client_object_ids')):
            x_px = bbox[0]
            y_px = bbox[1]
            xoffset_px = bbox[2]
            yoffset_px = bbox[3]

            # Save the detections to CSV

            detection = {
                        'id': str(uuid.uuid4()),
                        'obj_id': str(obj_id),
                        'class':   classname,
                        'score': float(score),
                        'x': int(x_px),
                        'y': int(y_px),
                        'x_offset': int(xoffset_px),
                        'y_offset': int(yoffset_px)}
            detection_list.append(detection)

    return detection_list


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

    with tf.gfile.GFile('{}/{}/frozen_inference_graph.pb'.format(path, model_name), 'rb') as fid:
        model_data = fid.read()

    sc = SparkContext('local', 'video_process_predict')
    model_data_bc = sc.broadcast(model_data)
    count = sc.parallelize(range(0, frame_count)).map(lambda x: extract_images(video_path, x))\
        .map(lambda x: predict(*x)).map(lambda x: process_result(*x)).filter(lambda x: print(x)).count()
    shutil.rmtree('{}/{}'.format(path, model_name))
