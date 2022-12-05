# JetBot_YOLO - Pose estimation

### 내 안의 작은 헬스 트레이너
>올바른 자세는 운동의 효율을 높여줍니다. 불필요한 개입을 방지하여 원하는 근육에 부하를 줄 수 있으며 피로를 최소화하고, 운동의 효과는 높게 유지할 수 있습니다. 따라서 Jetbot을 이용해 사용자의 운동 자세 예측 측정 프로그램을 개발하고자 합니다.

## 제작 동기
![제작동기_시장규모](https://user-images.githubusercontent.com/119550728/205535793-63cb4740-2b34-40bd-88d6-0a092011ec22.png)
>코로나19 바이러스가 전세계를 강타하면서 건강을 위해 운동을 하겠다는 인구가 높아지고 있습니다. 또한 다중이 이용하는 체육시설 대신 집에서 혼자 홈트레이닝을 하는 사람들이 많아졌는데 혼자 운동을 하다보니 정확한 자세로 운동을 하고 있는지 알기가 쉽지 않습니다. 안정적이지 못한 자세로 무리하게 운동을 하다보면 부상을 당할 수 있습니다.

![제작동기_위험성](https://user-images.githubusercontent.com/119550728/205544700-d34a524e-7927-48d8-a0b0-c341757a137b.png)
>실제로 잘못된 자세를 인지하지 못하고 계속해서 운동을 하다가 부상을 당한 사례도 많습니다.
>
>저희는 정확한 자세를 취했는지 알려주고, 관절에 어느정도의 무리가 가는지 체크해주는 프로그램을 만들어 올바른 자세를 통해 운동의 효과를 높여주고 부상의 위험성을 줄여주고자 합니다.



## Pose Track

Jetbot 내부에서 실시간 운동 자세 감지 추적하며 사용자에게 정보를 전달합니다.

>Detector는 Object detection과 Keypoint localization을 지원하는 keypoint 모델을 사용합니다.
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

