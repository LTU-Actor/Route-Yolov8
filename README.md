# ROS YOLOv8 Object Detection Package

This ROS package utilizes YOLOv8 for real-time object detection in images received from a camera topic. The package is designed to detect specific objects, such as stop signs, tires, and pedestrians, based on user requests.

## Prerequisites

- ROS (Robot Operating System)
- YOLOv8
- OpenCV
- Ultralytics
- EasyOCR

## Installation

1. Clone this repository to your ROS workspace's /src folder:

    ```bash
    git clone https://github.com/LTU-Actor/Route-Yolov8.git
    ```

2. Build the ROS package:

    ```bash
    cd <ros_workspace>
    catkin_make
    ```

## Usage

1. Launch the ROS node:

    ```bash
    roslaunch ltu_actor_route_yolov8_detector yolov8_detector.launch
    ```

2. Subscribe to the relevant topics to trigger object detection, set these parameters during launch:

    - Image Topic: `<imgtopic>`
    - Object Detection Topic: `<look_for_object_topic_name>` (string type, add unique strings and models in detect.py as needed)

3. View the results:

    The package provides detection results for stop signs, tires, and pedestrians. The detected objects, along with their sizes, are published to specific topics.

4. Debugging:

    If debugging is enabled (set in the rqt dynamic reconfigure), annotated images showing the detected objects are published to debug topics.

## Configuration

The package supports dynamic reconfiguration through the ROS Parameter Server. Parameters such as image resizing, flipping, and debugging can be adjusted on-the-fly.

Yolo v8 models can be added or swapped out in the `/models` folder and names updated in the launch file.

## Topics

- Inputs:
    - Image Topic: `<imgtopic_name>`
    - Input String Topic: `<look_for_object_topic_name>`

- Stop Sign Detection:
    - Detected Topic: `<stop_sign_detected_topic_name>`
    - Size Topic: `<stop_sign_size_topic_name>`
    - Debug Topic: `<stop_sign_debug_topic_name>`

- Person Detection:
    - Detected Topic: `<person_detected_topic_name>`
    - Size Topic: `<person_size_topic_name>`
    - Debug Topic: `<person_debug_topic_name>`

- Tire Detection:
    - Detected Topic: `<tire_detected_topic_name>`
    - Size Topic: `<tire_size_topic_name>`
    - Debug Topic: `<tire_debug_topic_name>`

## Contributing

Contributions are welcome!

## License

This project is licensed under the [MIT License](LICENSE).
