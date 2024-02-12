# YOLOBench

![YOLOBench](teaser.webp)
YOLOBench is a benchmarking python script for YOLO. , especially video image inference. It is designed to be easy to use and to provide a consistent and reproducible benchmarking environment. 

# Getting Started
YOLOBench uses Ultralytics' YOLOv8 as a backend. You need to install YOLOv8 first. 

## Setup Ultralytics' YOLOv8
 * ultralytics quick start https://docs.ultralytics.com/ja/quickstart/#install-ultralytics

## Clone YOLOBench
```
git clone https://github.com/TetsuakiBaba/YOLOBench.git
``` 

## Run YOLOBench
```
cd YOLOBench
python3 benchmark.py
```

# Benchmark Results

## yolov8 object detection inference [ms] (lower is better)
| ARCH | CPU/GPU | n | s | m | l | x |
| --- | --- | --- | --- | --- | --- | --- |
| Apple M1 | CPU | 60.3 | 115.3 | 217.6 | 381.9 | 547.9 |
| Apple M1 | GPU | 32.7 | 34.5 | 52.8 | 80.9 | 144.3 |
| Apple M1 MAX  | CPU | 36.3 | 55.9 | 98.7 | 161.3 | 223.5 |
| Apple M1 MAX  | GPU | 19.8 | 19.7 | 27.2 | 35.5 | 51.4 |
| Apple M2  | CPU | 39.0 | 72.8 | 145.8 | 234.0 | 332.7 |
| Apple M2  | GPU | 14.5 | 21.5 | 33.4 | 54.6 | 80.1 |
| Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz | CPU | 44.5 | 103.2 | 226.5 | 407.4 | 588.6 |
| NVIDIA RTX 1080Ti  | GPU | 9.6 | 9.3 | 12.7 | 18.3 | 24.8 |
| NVIDIA T4  | GPU | 9.2 | 9.6 | 15.0 | 23.8 | 36.7 |

## Thanks
  * by Free Videos: https://www.pexels.com/ja-jp/video/854100/