#!/bin/bash

echo "You should run this under sudo. This script doesn't check"
apt update
apt upgrade -y
pip install --user picamera[array]
echo "Enable camera in the below menu if not already configured"
sudo raspi-config # enable camera


apt install libatlas3-base libsz2 libharfbuzz0b libtiff5 libjasper1 libilmbase23 libopenexr23 libgstreamer1.0-0 libavcodec58 libavformat58 libavutil56 libswscale5 libqtgui4 libqt4-test libqtcore4 libwebp6 libhdf5-103 libhdf5-dev libatlas-base-dev  protobuf-compiler -y

echo "Downloading pip..."
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py

echo "Installing app requirements via pip"
pip3 install .
# Most dependencies should already be resolved, but just to be sure
pip3 install -r requirements.txt
# May require --ignore-installed passed to pip due to wrapt version
pip3 install opencv-python==3.4.6.27 tensorflow tf-slim
sudo apt-get install libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5 -y


echo "Downloading tensorflow models.."
git clone https://github.com/tensorflow/models
cd models/research/

echo "Compiling protobuf"
protoc object_detection/protos/*.proto --python_out=.
cd ../..
cp models/research/object_detection/packages/tf1/setup.py .

echo "export PYTHONPATH=$PYTHONPATH:/home/pi/models/research:/home/pi/models/research/slim" >> ~/.bashrcgru
