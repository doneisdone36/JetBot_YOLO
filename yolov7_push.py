import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
from yolov7_push_util import Angle
from PIL import Image, ImageDraw, ImageFont
import math

def IsOutlier_pushups(angle_list, joint):
  """
  한 프레임 차이인데 각도차이가 너무 많이 나면 그 값을 무시함.
  joint는 0, 1, 2 값을 가지며 각각 elbow, hip, knee를 나타냄
  """
  if len(angle_list) < 5:
    return False
  
  th = [35, 25]

  # 가장 최근 값 3개와 새로운 값이 차이가 너무 크면 outlier
  if abs((angle_list[-2]+angle_list[-3]+angle_list[-4])/3 - angle_list[-1]) > th[joint]:
      return True
  return False


@torch.no_grad()
def run(
        yolo_weights = 'yolov7-w6-pose.pt',
        source = 'test.mp4',
        device = 'cpu',
        tracker = True,
        drawskeleton = False
):      
    
    player = source.split('/')[-1].split('.')[-1].strip().lower()
    if player in ["mp4","webm","avi"] or player not in ["mp4","webm","avi"] and player.isnumeric():
        
        input_path = int(source) if source.isnumeric() else source 
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(yolo_weights,map_location=device)
        _ = model.eval()
        
        
        cap = cv2.VideoCapture(input_path)
        webcam = False
        
            
        fw, fh = int(cap.get(3)), int(cap.get(4))
        if player.isnumeric():
            webcam = True
            fw, fh = 640, 320
        
        vid_write_image = letterbox(
            cap.read()[1], (fw), stride=64,auto=True)[0]
        
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if source.isnumeric()else f"{input_path.split('/')[-1].split[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_kpt.mp4", cv2.VideoWriter_fourcc(*'mp4v'),30,(resize_width,resize_height))
        
        
        elbow_state = 'up'
        Is_Start = False
        temp = {'elbow_right':[],'knee_right':[]}
        pushups = []
        
        frame_count, total_fps = 0, 0
        
        bcount = 0
        direction = 0
        
        fontpath = "BMJUA_otf.otf"
        font = ImageFont.truetype(fontpath, 32)
        
        while True:
            ## yolov7 keypoint baseline start
            print(f"Frame {frame_count} Processing")
            
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
                
            ## yolov7 keypoint baseline end
                
                ## push up estimation
                
                im = Image.fromarray(img)
                draw = ImageDraw.Draw(im)
                draw.text((30,10), f"{len(pushups)}", font=font, fill=(0,0,0))
                img = np.array(im)
                
                if tracker :
                    for idx in range(output.shape[0]):
                        
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
                            
                        if not Is_Start :
                            temp['elbow_right'].pop()
                            temp['knee_right'].pop()
                        else :
                            if IsOutlier_pushups(temp['elbow_right'], 0) or IsOutlier_pushups(temp['knee_right'], 1) :
                                temp['elbow_right'].pop()
                                temp['knee_right'].pop()
                            else :  
                                if Is_elbow_up and  lower_body_condition :
                                    print("lower_body_condition")
                                    pushup_endtime = time.time()
                                    elbow_state = "up"
                                    element = []
                                    
                                    if max(temp['elbow_right']) > 160:
                                        element.append(1)
                                    else :
                                        element.append(0) 
                                        
                                    if min(temp['elbow_right']) < 70:
                                        element.append(1)
                                    else:
                                        element.append(0)
                                    
                                    temp['elbow_right'] = []
                                    
                                    if min(temp['knee_right']) < 130:
                                        element.append(0)
                                    else:
                                        element.append(1)
                                    
                                    temp['knee_right'] = []
                                    
                                    pushups.append(element)
                                
                                
                            
                                elif Is_elbow_down and elbow_state =='up' and lower_body_condition:
                                    pushup_starttime = time.time()
                                    elbow_state = 'down'
                            
                            
                            print("pushups",len(pushups))
                            
                                
                            
                    
                        percentage = np.interp(elbow_right_angle, (210,290), (0,100))
                        bar = np.interp(elbow_right_angle, (220,290), (int(fh)-100,100))
                        
                        
                        
                        
                        color = (254,118,136)
                        
                        if percentage == 100 :
                            if direction == 0 :
                                bcount += 0.5
                                direction += 1
                                
                        if percentage == 0:
                            if direction == 1:
                                bcount += 0.5
                                direction = 0
                                
                    
                if drawskeleton : 
                    for idx in range(output.shape[0]):
                        plot_skeleton_kpts(img, output[idx, 7:].T,3)
                
                if webcam : 
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break
                
                else : 
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960,540), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Detection", img)
                    cv2.waitKey(1)
                    
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                print(f'estimation:({end_time-start_time:.3f}s)')
                
            else:
                break
            
        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS : {avg_fps}")


                    
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)  
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    return opt
    
    
def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    
    