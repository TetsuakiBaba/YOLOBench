import cv2
from ultralytics import YOLO
import datetime
import time
import os
import sys

# Load the YOLOv8 models
modelnames = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
# modelsにmodelnamesのモデルを格納
models = [YOLO(modelname) for modelname in modelnames]
# models = [YOLO('yolov8n'), YOLO('yolov8s'), YOLO(
#     'yolov8m'), YOLO('yolov8l'), YOLO('yolov8x')]

# device = "cpu"
device = "mps"
# device = "cuda:1"

# Open the video file
video_path = 'sample.mp4'
# width = 640
# height = 384
# cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

loopnum = 0
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
    print(f"{modelnames[loopnum]} | {average_execution_time} ms")

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    loopnum += 1
