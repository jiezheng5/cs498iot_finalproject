# Introduction
This project is an [implementation of Lab 1: Devices](https://docs.google.com/document/d/1ppZn1diJxWYqtZe8EWygvLsDDvObp3NYrYPjHCb5JeY/edit#) for CS498 at UIUC.

Our report on the installation can be found [here](doc/report.pdf). This goes over the various high-level system connections.

Our video can be found on Mediaspace [here](https://mediaspace.illinois.edu/media/t/1_6bn3guc2)

# Running the system
## High-level Architecture
There are three main components to this system:
1) ADAS which captures images for object detection, makes braking decisions, and transmits information to the head unit. Found in [src/adas](src/adas)
2) Sensors + CAN-IP Gateway, this is used for tx/rx from and two distance and brake. Found in [src/can-ip-gateway-micro](src/can-ip-gateway-micro)
3) Dashboard, which displays the messages (including images with detected objects) from the ADAS. Found in [src/headunit](src/headunit)

## Setup
1. Interconnect the system per the diagram.
![wiring diagram](doc/wiring_diagram.png)
2. Setup the ADAS. This involves installing the needed dependencies. The [src/adas/config](src/adas/config) directory includes some automation steps for this process. More detailed instructions on the ADAS setup and configuration can be [found in the ADAS README](src/adas/README.md)
3. Flash the microcontrollers. The `combo.ino` file can be flashed directly, alternatively this project takes advantage of [Arduino-mk](https://github.com/sudar/Arduino-Makefile) which will need to be installed to run the included makefile. Confirm `isGateway` in combo.ino is set to 1 and flash the arduino acting as a gateway. Repeat this step for the sensor arduino but with isGateway set to 0.
4. On the laptop load the static HTML `src/headunit/dashboard.html` in a modern browser. This will not display anything until the pi server is started and the browser refreshed.
5. Run `python3 app.py` on the `/home/pi` directory of the ADAS. Wait until the message `Websocket Server has started` has printed.
6. Refresh the dashboard, wait approximately 10 seconds for the connection to initialize. When an image appears everything is up and running.

## Notes
[IoT_Lab_1_DistanceTest.pdf](doc/IoT_Lab_1_DistanceTest.pdf)
* test setup and expected results for distance readout from Arduino over the ethernet.

[test_distance.py](test/can-ip-gateway-micro/test_distance.py)
* `python3 test_distance.py` can help test the distance readout from Arduino over the ethernet. See the physical setup and expected results in doc/IoT Lab 1_DistanceTest.pdf.

[Wiring Diagram](doc/wiring_diagram.png)
* Explains the interconnections between components

[running_camera.y](test/adas/running_camera.py)
* Runs the camera on the pi directly for debugging and model confirmation

## References
See lab report