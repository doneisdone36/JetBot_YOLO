#install jupyterlab
#!/bin/bash

sudo apt-get install python-setuptools
sudo pip3 install setuptools
sudo pip3 install jupyterlab
jupyter lab --generate-config
cd /home/$USER/.jupyter
python3
from notebook.auth import passwd
passwd()
exit()
echo "c.NotebookApp.ip='0.0.0.0'\n
c.NotebookApp.port=8888\n
c.NotebookApp.open_browser=False\n
c.NotebookApp.notebook_dir=r'/home/[유저명]/workspace'\n
c.InlineBackend.figure_format=['svg','png']\n
c.NotebookApp.allow_origin = '*'\n
c.NotebookApp.password = '[sha1:암호화 패스워드]'\n" >> jupyter_notebook_config.py
mkdir ~/workspce
jupyter lab