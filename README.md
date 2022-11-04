# YOLO Webcam Demo

- This repo is a simple demo that perform object detection by using YOLOv3 (or YOLOv3-tiny, a lite version).

- To try my demo you need to clone this repo with:
  
    
    git clone https://github.com/ncl-crb/YOLO-demo.git

- After this, you need to download the weights of the network with:

      wget https://pjreddie.com/media/files/yolov3.weights 
  or 

      wget https://pjreddie.com/media/files/yolov3-tiny.weights 


  These are the official weights of YOLOv3 obtained by training the network on COCO dataset.
  You need to store those files in yolo_coco folder. <br><br>

- Now you can start to detect objects with:
    

       python webcam.py

- If you want to set different threshold, confidence or input image size you can type:


       python webcam.py --help

and set what you want. Note that decreasing or increasing the input size, you can respectively 
speed-up or speed-down the system.



## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```