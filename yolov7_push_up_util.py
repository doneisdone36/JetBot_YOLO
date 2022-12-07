import math
import cv2
import numpy as np

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
    
    
    # angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2,x1-x2))
    angle = calculate_angle2D(coords[p1][1:3],coords[p2][1:3],coords[p3][1:3])
    
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
        

        
        