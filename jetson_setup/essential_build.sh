#jetson_install_history
#first download jetson firmware https://developer.nvidia.com/embedded/l4t/r32_release_v7.1/jp_4.6.1_b110_sd_card/jeston_nano/jetson-nano-jp461-sd-card-image.zip
#!/bin/bash
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev
sudo pip3 install -y Cython numpy=='1.19.4'
#torch-1.9
wget https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl
sudo python3 -m pip install torch-1.9.0-cp36-cp36m-linux_aarch64.whl
#torch-vision
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch release/0.10 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.10.0
python3 setup.py install --user
cd ../
pip install 'pillow<7'
#installaion detectron prerequired
sudo apt-get install -y python-matplotlib python3-testresources
#yolov7
git clone -b v0.5 https://github.com/WongKinYiu/yolov7.git
#mv_files
cd yolov7
pip3 install -r "requirement.txt"
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt

#jtop
sudo -H pip3 install jetson-stats