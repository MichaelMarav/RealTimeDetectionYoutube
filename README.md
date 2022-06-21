# Real-time youtube video extraction and object detection/tracking 

This package contains the following two modules that can be utilized separately or combined.
1) Video extraction from youtube to OpenCV frames (works for live youtube live streams too)
2) Use the extracted OpenCV frames for object detection and tracking.

 
### Clone this repository 
```
$ git clone repository_name
```

# 1. Real-time youtube video extraction
This section describes how to extract a youtube video and convert it into OpenCV frames that you can work with (**FILENAME**). This code is tested on Ubuntu 18.04 and python3.6.9. In case you want to install the following packages for a specific python version, replace

```
$ pip install <package_name>
```
with 
```
$ pythonX.X -m pip install <package_name>
```


## Installation


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

# 2. Object detection and tracking


### Clone the darknet repository
```
$ git clone https://github.com/pjreddie/darknet
```
### Download weights
``` 
$ cd darknet/
$ mkdir weights/ && cd weights/
$ wget https://pjreddie.com/media/files/yolov3.weights
```



