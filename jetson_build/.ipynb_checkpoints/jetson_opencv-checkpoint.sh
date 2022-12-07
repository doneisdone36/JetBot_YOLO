#install jetson_opencv
#!/bin/bash

# check your memory first
$ free -m
# you need at least a total of 8.5 GB!
# if not, enlarge your swap space as explained in the guide 
$ wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-6-0.sh 
$ sudo chmod 755 ./OpenCV-4-6-0.sh 
$ ./OpenCV-4-6-0.sh 
# once the installation is done...
$ rm OpenCV-4-6-0.sh 