# See README.md for resources consulted as part of this portion of the lab.

# General imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
from datetime import datetime
import time

# PiCamera imports
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from picamera import PiCameraCircularIO

# Tensorflow imports
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Model related imports
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Ensure that model related resources are on the system path for pthon to find
PATH_TO_MODELS_RESEARCH = "/home/pi/models/research"
sys.path.append(PATH_TO_MODELS_RESEARCH)


# The model we're using for this object detection application.
# Chosen because it's very lightweight (as object detection models go) and is trained on low-res images.
#MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_NAME='ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'


# Download model if necessary. Note that this should be done as part of setup, but this is a relatively cheap check
# here in case it was not downloaded or put in the correct location.
if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve('http://download.tensorflow.org/models/object_detection/' + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Create the mapping from label index to human-readable object label (e.g. 'Person')  
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
#PATH_TO_LABELS='/home/pi/final_project/v1/fa20-cs498it-lab1-master/src/adas/object_detection/data/mscoco_label_map.pbtxt'
#categories = label_map_util.create_categories_from_labelmap(PATH_TO_LABELS, use_display_name=True

sys.path.append('')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

recording_folder='/home/pi/final_project/v1/fa20-cs498it-lab1-master/recordings'
if not os.path.exists(recording_folder):
    os.makedirs(recording_folder)

def run_inference_for_single_image(image, graph_ops, sess):
    '''
    Given a single image, represented as a numpy array, run object detection using the provided graph_ops and
    Tensorflow session, returning a dictionary containing details on detected objects (if any).
    :param image: The input image, given as a numpy array
    :param graph_ops: the operations (tensors) for the graph. Passed in simply so they don't have to be retrieved
                      for every image
    :param sess: the Tensorflow session
    :return: a dict containing arrays for each of the following TF outputs (with the output name as the key):
                ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']
    '''
    # Get handles to input and output tensors
    all_tensor_names = {output.name for op in graph_ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    if 'detection_masks' in tensor_dict:
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

class TensorCamera:
    """Wrapping in a class to capture the tensor and camera state in order to avoid
    some verbose functions or structures"""
    
    def __init__(self, width=640, height=480):
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.IM_WIDTH = width
        self.IMG_HEIGHT = height
        self.objectsOfInterest=['person','bird','cat','dog','horse','sheep','cow','bear','teddy bear']
        self.minScore=0.40
        ipv4 = os.popen('ip addr show eth0').read().split("inet ")[1].split("/")[0]
        self.camId=ipv4[-3:]
        self.imageSaveDeltaSeconds=10
        self.videoLoopSeconds=5
        self.videoPreRecordSeconds=1.5
        self.lastImageSaveTime=time.time()
        self.lastVideoSaveTime=time.time()
        self.saveVideoAtTime=time.time()
        self.videoLoopFlag=0 # used  to indicate that the current stream will need to be recorded
 
        # load in the graph and muddle with the default graph
        # this setup allows the same session to be used throughout
        # tf_session is *not* thread-safe
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
        self.tf_session = tf.Session(graph=self.graph)

        self.camera = PiCamera(resolution = (width, height), framerate = 30)
        
        #Brittany added for circular stream
        self.stream=PiCameraCircularIO(self.camera, seconds=5) #self.videoLoopSeconds)
        #self.camera.stop_recording()
        self.camera.start_recording(self.stream, format='h264')
        
        # Capture and set all to zero to get a blank array of the right size
        self._frame = PiRGBArray(self.camera, size=(width, height))
        self._frame.truncate(0)
        
    def capture_and_tell(self):
        """Uses the tensorflow graph to capture an image, do magic, ask the sensor
        board for the distance from an object, performs some business logic, and then
        send an update to the sensor board with brake status

        returns the system status for brake, distance, and image. This is to
        be used by a server component"""

        # clear the buffer frame used for capture
        self._frame.truncate(0)
        
        self.camera.capture(self._frame, format="rgb", use_video_port=True)
        frame = np.copy(self._frame.array)
        frame.setflags(write=1)

        graph_ops = self.graph.get_operations()
        with self.graph.as_default():
            output_dict = run_inference_for_single_image(frame, graph_ops, self.tf_session)
        
        # print(category_index)
        # print(output_dict['detection_scores'])
        # print(output_dict['detection_classes'])
        now=datetime.now()
        dateString=now.strftime("%Y%m%d%H%M%S")
        #timeString_return=now.strftime("%Y/%m/%d%H%M%S")
        detectionString=''
        for i,score in enumerate(output_dict['detection_scores']):
            if score >= self.minScore:
                detectionClass=output_dict['detection_classes'][i]
                dname=category_index[detectionClass]['name']
                print(dname, score)
                detectionString+=dname+'_'
        #print(dateString, detectionString)
        fname=dateString+detectionString+self.camId
        if detectionString:
            print(dateString, detectionString)
            #if enough time has elapsed save an image
            if time.time()-self.lastImageSaveTime>self.imageSaveDeltaSeconds:
                self.lastImageSaveTime=time.time()
                #save an image
                self.camera.capture(recording_folder+'/'+fname+'.jpg')                
                print('time to save an image add the code')
            #if the video loop flag is unset and enough time has passed set it
            #don't actually save the video here
            if not self.videoLoopFlag:
                if time.time()-self.lastVideoSaveTime>self.videoLoopSeconds:
                    self.videoLoopFlag=1
                    self.saveVideoAtTime=time.time()+(self.videoLoopSeconds-self.videoPreRecordSeconds)
                    self.videoName=dateString+detectionString+self.camId
        #lets save a video if it's time
        if self.videoLoopFlag and (time.time()>self.saveVideoAtTime):
            self.videoLoopFlag=0
            #camera.wait_recording(self.videoLoopSeconds)
            self.stream.copy_to(recording_folder+'/'+self.videoName+'.h264')
            print('time to save a video   ',self.videoName)                  
            
        
        # Draw labeled bounding boxes for any detected objects whose score is greater than 0.3
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=6,
            min_score_thresh=0.3)

        # Debug
        # from matplotlib import pyplot
        # pyplot.imshow(frame)
        # pyplot.savefig("test.png")


        # putting this in the main loop and requiring it for every frame
        # could faster, but this is low hanging fruit. Does not respond
        # well to network hiccups
 

        # Update the brake based on the detection and distance logic
        # send_brake does not send a signal to the light if not needed (i.e. no state change)
        # NOTE: Here we've used a static threshold for both object distance and the detection score for when to signal
        #       the brake because those have seemed to work well in our testing. Obviously we don't have a real vehicle
        #       to test this on, but if we did, then we could easily adjust this behavior accordingly. For example,
        #       we could make the score threshold based on the object distance, or store the last distance measurement
        #       or two (using variables on our TensorCamera class) to determine whether and how fast an object was
        #       approaching.
        brake='break string'
             
        self._frame.truncate(0)

        # Invert the frame to RGB and then encode it into
        # a png that can be transmitted as raw bytes to be displayed on the head unit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (_, buf) = cv2.imencode('.png', rgb_frame)


        if detectionString:
            break_string=detectionString.replace('_', ' ') 
        else:
            break_string='NA'

        # return the dashboard payload
        return {'image': buf.tobytes(),
                'distance': "",
                'brake': break_string}

