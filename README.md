# YOLO ML Project

1 - Clone this repo:

	git clone https://github.com/Murgia94/ML_YOLO1

2 - Download the weights of the network with:

	wget https://pjreddie.com/media/files/yolov3.weights 
	
	if you want to try YOLOv3
	
	wget https://pjreddie.com/media/files/yolov3-tiny.weights
	
	if you want to try YOLOv3_tiny

YOLOv3 official weights obtained by training the network on COCO dataset.
these files should be stored in the yolo_coco folder.

3 - For start to detect objects, open the command prompt in the reference folder and run:
	
	python YOLOml.py

If you want to set different thrershold, confidence or input image size you can run:

	python YOLOml.py --help




## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
