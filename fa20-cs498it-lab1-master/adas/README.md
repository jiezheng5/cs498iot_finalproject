# ADAS - Advanced Driver-Assistance System
The ADAS provides the main computation and logic controlling the "vehicle's" object detection and avoidance system.
Our ADAS is a Raspberry Pi, which (a) captures images using the attached PiCamera, (b) runs TensorFlow object detection
on that image, (c) retrieves distance information from ultrasonic sensor via a request to the CAN gateway,
and (d) combines that distance information with the object detection output to determine whether to 'hit the brakes' (by sending an appropriate packet to the brake light).
The output from each round of inference (i.e. image, objects detected if any, and distance) is also sent to the headunit.

## Setup
1. Flash the most recent version of Raspberry Pi OS.
2. Configure the IP address to 10.0.0.3/24, setup wifi (for package installation), and configure the camera. This can be done headlessly and is done in the `config` directory, but it's easier to complete these steps manually.
3. Copy the contents of this folder to /home/pi on the raspberry pi flashed and configured in the previous steps
4. Run the commands in setup_commands.sh. This command will prompt for camera configuration. You'll need to set this executable with `chmod +x setup_commands.sh`. You'll be prompted by `raspi-config` where you will need to enable the camera peripheral.  All python dependencies will be installed from those listed in `requirements.txt`. These can be directly installed by running `pip3 install -r requirements.txt`.
5. Move `object_detection` (a directory inside the TensorFlow `models` repo installed by `setup_commands.sh` from `research/models` to `/home/pi`
6. The entire contents of the `adas` folder should be placed at the `/home/pi` directory
7. Run the application with `python3 app.py`. This will start a websocket server on 10.0.0.3 which will send data to the dashboard when it connects. 


While these all should be installed under the local user. I did this all under root to avoid any potential permissions issues.


## Notes
1. There is a persistent error message from TensorFlow related to "HadoopFileSystem load error: libhdfs.so: cannot open shared object file: No such file or directory". While annoying, this does not affect the application.

## Resources Consulted / Citations

1. For using pre-trained models to perform object detection and draw the resulting bounding boxes on input images: https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/object_detection_tutorial.ipynb
2. Actual model used: http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09
3. For instructions on setting up PiCamera and (later) improving throughput: https://picamera.readthedocs.io/en/release-1.13/index.html
4. For initial Pi setup (i.e. which libraries and packages were needed to get the PiCamera and Tensorflow running)
and setting up a debugging pipeline for processed images before the headboard was integrated: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi
