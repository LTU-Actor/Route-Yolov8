# ROS YOLOv8 Object Detection Package

This ROS package utilizes YOLOv8 for real-time object detection in images received from a camera topic. The package is designed to detect specific objects, such as stop signs, tires, simulated potholes, and pedestrians, based on user requests. The package implements EasyOCR for differentiating between fake stop signs and real stop signs. The package can also display a mask to highlight an orange vest on a pedestrian.

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

    The package provides detection results for stop signs, tires, simulated potholes, and pedestrians. The detected objects, along with their sizes, are published to specific topics.

4. Debugging:

    If debugging is enabled (set in the rqt dynamic reconfigure), annotated images showing the detected objects are published to debug topics.

5. Vest Mask:

    If vest mask is enabled (set in the rqt dynamic reconfigure), an image showing the vest mask is published to the vest mask topic.

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

- Person Detection:
    - Detected Topic: `<person_detected_topic_name>`
    - Size Topic: `<person_size_topic_name>`

- SimulatedPothole Detection:
    - Detected Topic: `<pothole_detected_topic_name>`
    - Size Topic: `<pothole_size_topic_name>`

- Tire Detection:
    - Detected Topic: `<tire_detected_topic_name>`
    - Size Topic: `<tire_size_topic_name>`

- Debugging:
    - Object Debug Topic: `<object_debug_topic_name>`

- Vest Mask:
    - Vest Mask Topic: `<vest_mask_topic_name>`
  
## Contributing

Contributions are welcome!

## License

This project is licensed under the [MIT License](LICENSE).
