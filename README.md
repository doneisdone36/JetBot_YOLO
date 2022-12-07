# JetBot_YOLO - Pose estimation
<img src="https://img.shields.io/badge/Jetson Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white"><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

# Project
 - yolov7-w6-keypoint을 사용하여 사람의 몸을 확인 합니다.
 - yolo를 통해 가져온 값을 통해 사람을 측정합니다.
 
## Team

- 김규진
- 박태현
- 김연주


## Motivation
- 코로나19 바이러스가 전세계를 강타하면서 건강을 위해 운동을 하겠다는 인구가 높아지고 있습니다. 또한 다중이 이용하는 체육시설 대신 집에서 혼자 홈트레이닝을 하는 사람들이 많아졌는데 혼자 운동을 하다보니 정확한 자세로 운동을 하고 있는지 알기가 쉽지 않습니다. 안정적이지 못한 자세로 무리하게 운동을 하다보면 부상을 당할 수 있습니다.  


- 저희는 정확한 자세를 취했는지 알려주고, 관절에 어느정도의 무리가 가는지 체크해주는 프로그램을 만들어 올바른 자세를 통해 운동의 효과를 높여주고 부상의 위험성을 줄여주고자 합니다.


# Project
yolov7-w6-keypoint 모델을 key-point tracking 하여 사용자의 운동 부하와 갯수를 측정합니다 

## Demo
<img width="250" alt="Screenshot%202022-11-29%20at%2011 27 18%20AM" src="https://user-images.githubusercontent.com/71868697/204456043-32986f4f-8340-4701-a800-d872a71bea59.png">

## Installation
```
$ git clone https://github.com/doneisdone36/JetBot_YOLO.git
$ cd JetBot_YOLO  
# Model download
$ sh jetson_setup/opencv.sh
$ sh jetson_setup/essential_build.sh

```

## Getting Started
```
$ python yolov7_push_up.py --source 0  # webcam
```

## Goal
1. 운동 콘텐츠를 접목시켜 콘텐츠 속 운동 동작과 현재 자신의 동작이 얼마나 일치하는지 체크해준다.
2. 바른 자세는 녹색, 틀린 자세는 적색으로 표시하여 사용자가 어느 부위의 신체가 바른 자세를 유지하고 있고, 어느 부위의 신체가 틀린 자세를 취하고 있는지 한눈에 알 수 있도록 해준다.
3. 운동 동작 영상과 분석 정보를 저장하고 축적해 사용자의 운동량을 체크해주고, 정확한 자세 정보에 대한 데이터를 보여준다.
