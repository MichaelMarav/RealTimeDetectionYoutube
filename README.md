# Real-time youtube video stream extraction and object detection 

This package contains two modules that perform real-time object detection from Youtube video stream. A possible use case is detection with a drone's camera since most of them support Youtube live-streaming (with some constant delay ~ 7secs). It can be also used with simple youtube videos just by providing the URL. 

The *SSD_youtube.py* uses the Single Shot Detection which runs on CPU i5 8400, with ~ 30 hz. The *YOLO_Youtube.py*, uses the YOLO algorithm and runs on the same CPU with ~1-3 hz. If you want high quality results, setup CUDA, CUDA-toolkit and CUDNN for opencv and use the YOLO detection algorithm.   
 
### Dependencies
1) Python3
2) OpenCV
3) Pafy 
4) Youtube-dl

# Installation
This code is tested on Ubuntu 18.04 with python3.6 and Ubuntu 20.04 with python3.8. In case you want to install the following packages for a specific python version, replace

```
$ pip install <package_name>
```
with 
```
$ pythonX.X -m pip install <package_name>
```

1. Install OpenCV 
```bash
$ pip install opencv-python
```
2. Install pafy and youtube-dl
```bash
$ pip install pafy 
$ pip install youtube-dl
```
Because youtube has removed the dislike count, you will get an error later on, when the library tries to extract the dislike counts of the video. To fix this:
1. Edit ~/.local/lib/python3.6/site-packages/pafy/backend_youtube_dl.py
2. Comment out line 54:
```python
#self._dislikes = self._ydl_info['dislike_count']
```

To subscribe to live-stream there is a method called **.getbest()** which grabs the highest possible quality of the stream. In case you want to increase performance you can go to 

*/home/michael/.local/lib/python3.6/site-packages/pafy/backend_shared.py*

line 359 and change:
```python
# r = max(streams, key = _sortkey)
r = streams[i]
```
Choose a value for *i = 0,...,size(streams)* is a different quality with streams[-1] being the max possible quality and streams[0] the lowest.

# Setup YOLO detection

### Download YOLO weights
``` 
$ wget https://pjreddie.com/media/files/yolov3.weights
```

### Download coco.names:
https://github.com/pjreddie/darknet/blob/master/data/coco.names

### Download yolov3.cfg
https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

### Change the parameters 
*url* : The url you wish to subscribe from Youtube \
*abs_path_labels* : The absolute path for *coco.names* \
*abs_path_weights* : The absolute path for *yolov3.weights*\
*abs_path_config* : The absolute path for *yolov3.cfg*

Ready to go! Run the detection algorithm:
```
$ pythonX.X YOLO_Youtube.py
```
# 3. Run it on GPU 

Requirements (OpenCV > 4.2)


### Clone this repository 
```
$ git clone https://github.com/MichaelMarav/YoutubeRealTimeYOLO.git
```
