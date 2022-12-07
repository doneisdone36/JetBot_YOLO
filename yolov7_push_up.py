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


@torch.no_grad()
def run(
        yolo_weights = 'yolov7-w6-pose.pt',
        source = '0',
        device = 'cpu',
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
        
        vid_write_image = letterbox(
            cap.read()[1], (fw), stride=64,auto=True)[0]
        
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if source.isnumeric()else f"{input_path.split('/')[-1].split[0]}"
        # out = cv2.VideoWriter(f"{out_video_name}_kpt.mp4", cv2.VideoWriter_fourcc(*'mp4v'),30,(fw,fh))
        
        frame_count, total_fps = 0, 0
        
        while True:
            frame_count += 1 
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
                
                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(img, output[idx,7:].T,3)
            
                cv2.imshow('output',img)
                cv2.waitKey(1)  # 1 millisecond

                
                    
                
            
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    return opt
    
    
def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    
    