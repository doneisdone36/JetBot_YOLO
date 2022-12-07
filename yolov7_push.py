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
from trainer import findAngle
from PIL import Image, ImageDraw, ImageFont
import math

@torch.no_grad()
def run(
        yolo_weights = 'yolov7-w6-pose.pt',
        source = 'test.mp4',
        device = 'cpu',
        curl_traker = True,
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
            fw, fh = 1280, 768
        
        # vid_write_image = letterbox(
        #     cap.read()[1], (fw), stride=64,auto=True)[0]
        
        # resize_height, resize_width = vid_write_image.shape[:2]
        # out_video_name = "output" if source.isnumeric()else f"{input_path.split('/')[-1].split[0]}"
        # out = cv2.VideoWriter(f"{out_video_name}_kpt.mp4", cv2.VideoWriter_fourcc(*'mp4v'),30,(resize_width,resize_height))
        
        # 
        
        
        frame_count, total_fps = 0, 0
        
        bcount = 0
        direction = 0
        
        
        fontpath = "BMJUA_otf.otf"
        font = ImageFont.truetype(fontpath, 32)
        font1 = ImageFont.truetype(fontpath, 32)
        
        
        while True:
            # frame_count += 1 
            print(f"Frame {frame_count} Processing")
            ret,frame = cap.read()
            if ret:
                orig_frame = frame
                
                image = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
                
                
                if webcam :
                    image = cv2.resize(image, (fw,fh), interpolation=cv2.INTER_LINEAR)
                    
                image = letterbox(image, (fw), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                
                image = image.to(device)
                image = image.float()
                start_time = time.time()
                
                
                output, _ = model(image)
                    
                output = non_max_suppression_kpt(
                    output, 0.5, 0.65, nc = model.yaml['nc'], nkpt = model.yaml['nkpt'], kpt_label=True
                )
                
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # for idx in range(output.shape[0]):
                #     plot_skeleton_kpts(img, output[idx,7:].T,3)
            
                # cv2.imshow('output',img)
                # cv2.waitKey(1)  # 1 millisecond
                

                if curl_traker :
                    for idx in range(output.shape[0]):
                        print("len kpts :",len(output[idx, 7:].T))
                        kpts = output[idx, :7].T
                        print("len kpts2 :",len(kpts))
                        print("kpts :",output[idx, 7:].T)
                        
                        angleRight = findAngle(img, output[idx, 7:].T,5,7,9, draw=True)
                        
                        
                        percentage = np.interp(angleRight, (210,290), (0,100))
                        bar = np.interp(angleRight, (220,290), (int(fh)-100,100))
                        
                        angleLeft = findAngle(img, output[idx, 7:].T,6,8,10, draw=True)
                        
                        
                        color = (254,118,136)
                        
                        if percentage == 100 :
                            if direction == 0 :
                                bcount += 0.5
                                direction += 1
                                
                        if percentage == 0:
                            if direction == 1:
                                bcount += 0.5
                                direction = 0
                                
                        cv2.line(img,(100,100),(100, int(fh)-100),(255,255,255), 30)
                        cv2.line(img,(100,int(bar)),(100, int(fh)-100), color, 30)
                        
                        if (int(percentage) < 10):
                            cv2.line(img, (155, int(bar)), (190,int(bar)),color,40)
                        elif((int(percentage) >= 10) and (int(percentage)>100)):
                            cv2.line(img, (155, int(bar)), (200,int(bar)),color,40)
                        else:
                            cv2.line(img, (155, int(bar)), (210,int(bar)),color,40)
                            
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.rounded_rectangle((fw-300, (fh//2)- 100, fw-50, (fh//2) * 100),fill = color, radius= 40)

                        draw.text(
                            (145,100), f"{int(percentage)}%", font=font, fill=(255,255,255))
                    
                        draw.text(
                            (fw-230,(fh//2)-100), f"{int(bcount)}%", font=font1, fill=(255,255,255))

                        img = np.array(im)
                    
                if drawskeleton : 
                    for idx in range(output.shape[0]):
                        print("len kpts :",len(output[idx, 7:].T))
                        print("kpts :",output[idx, 7:].T)
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
                # out.write(img)
            else:
                break
            
        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS : {avg_fps:.3f}")
                    
def findAngle(image, kpts, p1,p2,p3, draw = True):
    coords = []
    no_kpts = len(kpts)//3
    for i in range(no_kpts):
        cx,cy = kpts[3*i], kpts[3*i + 1]
        conf = kpts[3*i + 2]
        coords.append([i, cx,cy, conf])
        
    points = (p1,p2,p3)
    print("p1 : ",p1)
    print("p2 : ",p2)
    print("p3 : ",p3)
    print("coords shape: ",len(coords))
    
    x1,y1 = coords[p1][1:3]
    x2,y2 = coords[p2][1:3]
    x3,y3 = coords[p3][1:3]
    
    
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2,x1-x2))
    
    if angle < 0:
        angle += 360
        
    
    if draw : 
        cv2.line(image, (int(x1),int(y1)), (int(x2),int(y2)),(255,255,255),3)
        cv2.line(image, (int(x3),int(y3)), (int(x2),int(y2)),(255,255,255),3)
        
        
        cv2.circle(image, (int(x1),int(y1)),10 ,(255,255,255),cv2.FILLED)
        cv2.circle(image, (int(x1),int(y1)),10 ,(255,255,255),cv2.FILLED)
        cv2.circle(image, (int(x1),int(y1)),10 ,(255,255,255),cv2.FILLED)
        
        cv2.circle(image, (int(x2),int(y2)),10 ,(255,255,255),cv2.FILLED)
        cv2.circle(image, (int(x2),int(y2)),10 ,(255,255,255),cv2.FILLED)
        cv2.circle(image, (int(x2),int(y2)),10 ,(255,255,255),cv2.FILLED)
        
        cv2.circle(image, (int(x3),int(y3)),10 ,(255,255,255),cv2.FILLED)
        cv2.circle(image, (int(x3),int(y3)),10 ,(255,255,255),cv2.FILLED)
        cv2.circle(image, (int(x3),int(y3)),10 ,(255,255,255),cv2.FILLED)
        
    return angle


def calculate_angle2D(a,b,c,direction):
  """
  calculate_angle2D is divided by left and right side because this function uses external product
  input : a,b,c -> landmarks with shape [x,y,z,visibility]
          direction -> int -1 or 1
                      -1 means Video(photo) for a person's left side and 1 means Video(photo) for a person's right side
  output : angle between vector ba and bc with range 0~360
  """
  # external product's z value
  external_z = (b[0]-a[0])*(b[1]-c[1]) - (b[1]-a[1])*(b[0]-c[0])

  a = np.array(a[:2]) #first
  b = np.array(b[:2]) #mid
  c = np.array(c[:2]) #end

  ba = b-a
  bc = b-c
  dot_result = np.dot(ba, bc)


  ba_size = np.linalg.norm(ba)
  bc_size = np.linalg.norm(bc)
  radi = np.arccos(dot_result / (ba_size*bc_size))
  angle = np.abs(radi*180.0/np.pi)

  if external_z * direction > 0:
    angle = 360 - angle

  return angle

                        
                
                    
                
            
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
    
    