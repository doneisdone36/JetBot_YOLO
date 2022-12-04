# JetBot_YOLO - Pose estimation

### 내 안의 작은 헬스 트레이너
>올바른 자세는 운동의 효율을 높여줍니다. 불필요한 개입을 방지하여 원하는 근육에 부하를 줄 수 있으며 피로를 최소화하고, 운동의 효과는 높게 유지할 수 있습니다. 따라서 Jetbot을 이용해 사용자의 운동 자세 예측 측정 프로그램을 개발하고자 합니다.



## Pose Track

Jetbot 내부에서 실시간 운동 자세 감지 추적하며 사용자에게 정보를 전달합니다.

>Detector는 Object detection과 Keypoint localization을 지원하는 keypoint 모델을 사용합니다
>  
>  
>Tracker는 Occlustion, ID Swtiching에 이점을 보이는 Detectron,Strong_SORT를 사용합니다.


### Getting Started
```
$ python demo.py --config-file [model path] --video-input [source] --confidence-threshold 0.6 --output [output]
```

### Detectron2
<img width="451" alt="Screenshot%202022-11-29%20at%2011 27 18%20AM" src="https://user-images.githubusercontent.com/71868697/204456043-32986f4f-8340-4701-a800-d872a71bea59.png">

### YOLO v7 + Strong_SORT

