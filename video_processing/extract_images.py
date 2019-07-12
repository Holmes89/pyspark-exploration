from pyspark import SparkContext
import cv2
import argparse


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


def save_image_file(index, image, dir):
    cv2.imwrite("{}/frame{}.png".format(dir, index), image)
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process video to create images')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('output', type=str, help='Path to directory to place images')

    args = parser.parse_args()

    video_path = args.video
    dir_out = args.output

    frame_count = get_frame_count(video_path)

    sc = SparkContext('local', 'video_process_extract_images')
    count = sc.parallelize(range(0, frame_count)).map(lambda x: extract_images(video_path, x)).filter(lambda x: save_image_file(*x, dir_out)).count()
