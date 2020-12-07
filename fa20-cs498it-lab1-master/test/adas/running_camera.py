## SRC: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
## SRC: https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/master/Object_detection_picamera.py

## Impprts
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from distutils.version import StrictVersion


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# This is needed since the notebook is stored in the object_detection folder.
PATH_TO_MODELS_RESEARCH = "/home/pi/models/research"
sys.path.append(PATH_TO_MODELS_RESEARCH)


## Load model if necessary
# What model to download.
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')

if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def run_inference_for_single_image(image, graph_ops, sess):
    # Get handles to input and output tensors
    all_tensor_names = {output.name for op in graph_ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def run_camera():
    import cv2
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    IM_WIDTH = 640
    IM_HEIGHT = 480

    with detection_graph.as_default():
        with tf.Session() as sess:
            frame_rate_calc = 1
            freq = cv2.getTickFrequency()
            font = cv2.FONT_HERSHEY_SIMPLEX
            camera = PiCamera()
            camera.resolution = (IM_WIDTH, IM_HEIGHT)
            camera.framerate = 2
            rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
            rawCapture.truncate(0)

            for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

                t1 = cv2.getTickCount()

                # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
                # i.e. a single-column array, where each item in the column has the pixel RGB value
                frame = np.copy(frame1.array)
                frame.setflags(write=1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame_expanded = np.expand_dims(frame_rgb, axis=0)
                graph_ops = tf.get_default_graph().get_operations()

                output_dict = run_inference_for_single_image(frame, graph_ops, sess)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.3)

                cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)

                # All the results have been drawn on the frame, so it's time to display it.
                cv2.imshow('Object detector', frame)

                t2 = cv2.getTickCount()
                time1 = (t2 - t1) / freq
                frame_rate_calc = 1 / time1

                #TODO: read distance
                distance = np.random.random() * 200

                if distance < 100 and any(output_dict['detection_scores']) > 0.3:
                    brake = True
                else:
                    brake = False


                res_json = {'image': frame,
                            'frame_rate': frame_rate_calc,
                            'distance': str(distance) + " cm",
                            'brake': brake}

                # TODO: Send res_json

                # TODO: Remove
                res_json.pop('image')
                print(str(res_json))
                print(str(output_dict['detection_scores']) + "\n\n")

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break

                rawCapture.truncate(0)

            camera.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
