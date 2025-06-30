import cv2
from ultralytics import YOLO
import datetime
import time
import os
import sys
import argparse
import platform
import subprocess

# Parse command line arguments
parser = argparse.ArgumentParser(description='YOLO Benchmark')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to run inference on (default: cpu). Options: cpu, mps, cuda:0, cuda:1, etc.')
args = parser.parse_args()

# Function to get system information


def get_system_info():
    system_info = {}

    # Get OS and architecture
    system_info['os'] = platform.system()
    system_info['arch'] = platform.machine()

    # Get CPU information and Apple Silicon details
    if system_info['os'] == 'Darwin':  # macOS
        try:
            cpu_info = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
            system_info['cpu'] = cpu_info

            # Get Apple Silicon chip information
            try:
                hw_model = subprocess.check_output(
                    ['sysctl', '-n', 'hw.model']).decode().strip()
                system_info['hw_model'] = hw_model
            except:
                system_info['hw_model'] = 'Unknown'

        except:
            system_info['cpu'] = 'Unknown CPU'
            system_info['hw_model'] = 'Unknown'
    elif system_info['os'] == 'Linux':
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        system_info['cpu'] = line.split(':')[1].strip()
                        break
        except:
            system_info['cpu'] = 'Unknown CPU'
        system_info['hw_model'] = 'Unknown'
    else:
        system_info['cpu'] = 'Unknown CPU'
        system_info['hw_model'] = 'Unknown'

    # Get Apple Silicon chip name from CPU info or hw.model
    def get_apple_silicon_name():
        cpu_lower = system_info['cpu'].lower()
        hw_model_lower = system_info.get('hw_model', '').lower()

        # Check for M4 variants
        if 'm4' in cpu_lower or 'm4' in hw_model_lower:
            if 'max' in cpu_lower or 'max' in hw_model_lower:
                return 'Apple M4 Max'
            elif 'pro' in cpu_lower or 'pro' in hw_model_lower:
                return 'Apple M4 Pro'
            else:
                return 'Apple M4'
        # Check for M3 variants
        elif 'm3' in cpu_lower or 'm3' in hw_model_lower:
            if 'max' in cpu_lower or 'max' in hw_model_lower:
                return 'Apple M3 Max'
            elif 'pro' in cpu_lower or 'pro' in hw_model_lower:
                return 'Apple M3 Pro'
            else:
                return 'Apple M3'
        # Check for M2 variants
        elif 'm2' in cpu_lower or 'm2' in hw_model_lower:
            if 'max' in cpu_lower or 'max' in hw_model_lower:
                return 'Apple M2 Max'
            elif 'pro' in cpu_lower or 'pro' in hw_model_lower:
                return 'Apple M2 Pro'
            elif 'ultra' in cpu_lower or 'ultra' in hw_model_lower:
                return 'Apple M2 Ultra'
            else:
                return 'Apple M2'
        # Check for M1 variants
        elif 'm1' in cpu_lower or 'm1' in hw_model_lower:
            if 'max' in cpu_lower or 'max' in hw_model_lower:
                return 'Apple M1 Max'
            elif 'pro' in cpu_lower or 'pro' in hw_model_lower:
                return 'Apple M1 Pro'
            elif 'ultra' in cpu_lower or 'ultra' in hw_model_lower:
                return 'Apple M1 Ultra'
            else:
                return 'Apple M1'
        else:
            return 'Apple Silicon'

    # Get GPU information based on device
    if args.device == 'mps':
        system_info['device_type'] = 'GPU'
        if system_info['os'] == 'Darwin':
            system_info['device_name'] = get_apple_silicon_name()
        else:
            system_info['device_name'] = 'Apple Silicon GPU'
    elif args.device.startswith('cuda'):
        system_info['device_type'] = 'GPU'
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(
                    int(args.device.split(':')[1]) if ':' in args.device else 0)
                system_info['device_name'] = gpu_name
            else:
                system_info['device_name'] = 'CUDA GPU'
        except:
            system_info['device_name'] = 'CUDA GPU'
    else:
        system_info['device_type'] = 'CPU'
        if system_info['os'] == 'Darwin':
            system_info['device_name'] = get_apple_silicon_name()
        else:
            system_info['device_name'] = system_info['cpu']

    return system_info


# Get system information
sys_info = get_system_info()

# Load the YOLOv8 models
modelnames = [
    'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
    'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'
]
# modelsにmodelnamesのモデルを格納
models = [YOLO(modelname) for modelname in modelnames]
# models = [YOLO('yolov8n'), YOLO('yolov8s'), YOLO(
#     'yolov8m'), YOLO('yolov8l'), YOLO('yolov8x')]

# device = "cpu"
device = args.device
# device = "cuda:1"

# Open the video file
video_path = 'sample.mp4'
# width = 640
# height = 384
# cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

loopnum = 0
results_dict = {}  # Store results for table output

for model in models:
    execution_times = []  # List to store execution times
    # Reset frame count and reopen the video file
    frame_count = 0
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # 現在時刻を記録
        start_time = time.time()

        # フレームを読み込む
        ret, frame = cap.read()
        if not ret:
            break

        # Read a frame from the video
        success, frame = cap.read()

        if success:
            frame_count += 1

            # Run YOLOv8 inference on the frame
            start_time = time.time()  # Start measuring the execution time

            results = model.predict(frame, device=device, verbose=False, show=False,
                                    show_labels=False, show_conf=False)

            end_time = time.time()  # End measuring the execution time
            # Calculate the execution time in milliseconds
            execution_time_ms = (end_time - start_time) * 1000

            # ignore the first 30 frames due to the warmup time
            if frame_count > 30:
                # Add execution time to the list
                execution_times.append(execution_time_ms)
                # print(f"Execution time: {execution_time_ms} ms")
            # results = model(frame, device="cpu")

            # print(len(results[0].boxes))

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # View results

            # Display the annotated frame
            cv2.imshow("YoloBenchmarks", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Calculate the average execution time
    average_execution_time = sum(execution_times) / len(execution_times)
    # print('execution_times:', execution_times)
    print(f"{modelnames[loopnum]} | {average_execution_time:.2f} ms")

    # Store result for table output
    results_dict[modelnames[loopnum]] = round(average_execution_time, 2)

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    loopnum += 1

# Print README-ready table format
print("\n" + "="*60)
print("README.md ready format:")
print("="*60)

# Create architecture description
if sys_info['device_type'] == 'CPU':
    arch_desc = sys_info['cpu']
    device_desc = 'CPU'
else:
    arch_desc = sys_info['device_name']
    device_desc = 'GPU'

# Print table header (only if this is the first result you're adding)
print("\nAdd this row to your README.md benchmark table:")
print(f"| {arch_desc} | {device_desc} | {results_dict.get('yolov8n', 'N/A')} | {results_dict.get('yolov8s', 'N/A')} | {results_dict.get('yolov8m', 'N/A')} | {results_dict.get('yolov8l', 'N/A')} | {results_dict.get('yolov8x', 'N/A')} |")

# Also print YOLO11 results if available
yolo11_results = [results_dict.get(
    f'yolo11{size}', 'N/A') for size in ['n', 's', 'm', 'l', 'x']]
if any(result != 'N/A' for result in yolo11_results):
    print(f"\nYOLO11 results:")
    print(
        f"| {arch_desc} | {device_desc} | {yolo11_results[0]} | {yolo11_results[1]} | {yolo11_results[2]} | {yolo11_results[3]} | {yolo11_results[4]} |")

print("\n" + "="*60)
