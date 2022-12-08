# JetBot_YOLO - Pose estimation
<img src="https://img.shields.io/badge/Jetson Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white"><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

# Project
 - yolov7-w6-keypoint을 사용하여 사람의 몸을 확인 합니다.
 - yolo를 통해 가져온 값을 통해 push-up을 측정합니다.
 
## Demo
<img src="./demo.gif" width="800" height="450"/>

## Team

- 김규진
- 박태현
- 김연주


## Motivation
- 코로나19 바이러스가 전세계를 강타하면서 건강을 위해 운동을 하겠다는 인구가 높아지고 있습니다. 또한 다중이 이용하는 체육시설 대신 집에서 혼자 홈트레이닝을 하는 사람들이 많아졌는데 혼자 운동을 하다보니 정확한 자세로 운동을 하고 있는지 알기가 쉽지 않습니다. 안정적이지 못한 자세로 무리하게 운동을 하다보면 부상을 당할 수 있습니다.  


- 저희는 정확한 자세를 취했는지 알려주고, 관절에 어느정도의 무리가 가는지 체크해주는 프로그램을 만들어 올바른 자세를 통해 운동의 효과를 높여주고 부상의 위험성을 줄여주고자 합니다.



## Installation
1. first download jetpack 4.6.1
 - [Offical_Downlaod_jetpack_4.6.1](https://developer.nvidia.com/embedded/l4t/r32_release_v7.1/jp_4.6.1_b110_sd_card/jeston_nano/jetson-nano-jp461-sd-card-image.zip)
2. Making bootable device 
 - [Offical_balena](https://www.balena.io/etcher/)
 - [Help Link](https://www.balena.io/blog/getting-started-with-the-nvidia-jetson-nano-using-balena/) 

```
$ git clone https://github.com/doneisdone36/JetBot_YOLO.git
$ cd JetBot_YOLO  
$ sh jetson_setup/opencv.sh # almost 3 hours
$ sh jetson_setup/essential_build.sh # Our files move into yolov7[WongKinYiu](https://github.com/WongKinYiu/yolov7) folders

```

## Packages
 - Jetpack 4.6.1
 - Python 3.6
 - Pytorch 1.9.0
 - Torch-vision 0.10.0
 - OpenCV 4.6.0
 - Numpy, pandas, seaborn, matplotliab ...

## Getting Started
```
$ python3 yolov7_push_up.py --source 0 [ --drawskeleton -Tracker ] # webcam
```

## Goal
1. 운동 콘텐츠를 접목시켜 콘텐츠 속 운동 동작과 현재 자신의 동작이 얼마나 일치하는지 체크해준다.
2. 바른 자세는 녹색, 틀린 자세는 적색으로 표시하여 사용자가 어느 부위의 신체가 바른 자세를 유지하고 있고, 어느 부위의 신체가 틀린 자세를 취하고 있는지 한눈에 알 수 있도록 해준다.
3. 운동 동작 영상과 분석 정보를 저장하고 축적해 사용자의 운동량을 체크해주고, 정확한 자세 정보에 대한 데이터를 보여준다.

## Counting push-up
<img src="./push_up_analysis.png" width="800" height="450"/>

## Code Block
 load image to cv2 [yolov7-w6-pose baseline](https://github.com/WongKinYiu/yolov7/blob/main/tools/instance.ipynb)
```python
# yolov7_push.py
ret,frame = cap.read()
if ret:
    start_time = time.time()
    orig_frame = frame
    image = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    
    if webcam :
        image = cv2.resize(image, (fw,fh), interpolation=cv2.INTER_LINEAR)
        
    image = letterbox(image, (fw), stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    
    image = image.to(device)
    image = image.float()
    
    output, _ = model(image)
        
    output = non_max_suppression_kpt(
        output, 0.5, 0.65, nc = model.yaml['nc'], nkpt = model.yaml['nkpt'], kpt_label=True
    )
    
    output = output_to_keypoint(output)
    img = image[0].permute(1, 2, 0) * 255
    img = img.cpu().numpy().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
```
Push up Angle
```python
## estimate elbow up and down
elbow_right_angle = Angle(img, output[idx, 7:].T,5,7,9, draw=True)
Is_elbow_up = elbow_right_angle > 130
Is_elbow_down = elbow_right_angle < 110
temp['elbow_right'].append(elbow_right_angle)
print(f'elbow angle {elbow_right_angle}')

## estimate knee up and down
knee_right_angle = Angle(img, output[idx, 7:].T,11,13,15, draw=True)
knee_condition = (knee_right_angle > 130) and (knee_right_angle < 220)
temp['knee_right'].append(knee_right_angle)
print(f'knee angle : {knee_right_angle}')


lower_body_condition =  knee_condition

Is_pushup_angle = (elbow_right_angle > 140)  and  (knee_right_angle > 125)  
if Is_pushup_angle :
    Is_Start = True
    print("success")

```
getAngle
```python
def Angle(image, kpts, p1,p2,p3, draw = True):
    coords = []
    no_kpts = len(kpts)//3
    for i in range(no_kpts):
        cx,cy = kpts[3*i], kpts[3*i + 1]
        conf = kpts[3*i + 2]
        coords.append([i, cx,cy, conf])
        
    points = (p1,p2,p3)

    
    x1,y1 = coords[p1][1:3]
    x2,y2 = coords[p2][1:3]
    x3,y3 = coords[p3][1:3]
    
    
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2,x1-x2))

```
## Version_Log
branch
 - jetson_install -> jetson_setup(0.0.4)
 - estimation -> estimate person push-up(latest: 0.0.4)

