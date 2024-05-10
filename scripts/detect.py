#!/usr/bin/env python3
########################################################################
import gc  # Garbage collection
import time

import cv2  # OpenCV version 2.x (ROS limitation)
import easyocr  # OCR
import numpy as np  # numpy
import rospy  # ROS API
import torch
import ultralytics  # Yolo v8
from cv_bridge import CvBridge, CvBridgeError  # ROS to/from OpenCV API
from dynamic_reconfigure.server import Server  # ROS Parameter server for debug
from ltu_actor_route_yolov8_detector.cfg import YoloDetectConfig  # packageName.cfg
from sensor_msgs.msg import Image  # Image message
from std_msgs.msg import (  # UInt8(used as bool) and UInt32(used as +ve number) message
    String,
    UInt8,
    UInt32,
)

########################################################################
### Global Variables:
global config_  # Dynamic reconfiguration holder
global bridge  # ROS-CV bridge
bridge = CvBridge()

# >>> INITIALIZE MODEL PATHS HERE: <<<
global model_stop_path  # Get yolov8 stop sign detection model's path
global model_coco_path  # Get yolov8 COCO trained model's path
global model_tire_path  # Get yolov8 tire detection model's path
global model_u_model_path  # Get yolov8 u_model path

global display_size
display_size = 640  # pixel resolution used for debug outputs

global image_size  # pixel resolution used for inference
global cam_image  # image used for detection

global yolo_called
yolo_called = False  # True if yolo detection is called for from topic
global real_stop_sign_detected
real_stop_sign_detected = False  # True if real stop sign has been detected
# OCR used to extract text from the image
global ocr_reader
ocr_reader = easyocr.Reader(
            ["en"], gpu=True
        )  # this needs to run only once to load the model into memory

########################################################################
### Functions:

# This is a comment


# Image callback - Converts ROS Image to OpenCV Image
def get_image_callback(Image):
    # if Yolo is enabled from Dynamic reconfigure
    if config_.enable:
        global bridge, cam_image, image_size
        try:  # convert ros_image into an opencv-compatible image
            cv_image = bridge.imgmsg_to_cv2(Image, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Now cv_image is a standard OpenCV matrice ---------------------------------------

        # Resize according to dynamic reconfigure settings
        cv_image = resize_image(
            cv_image, size=(config_.image_resize, config_.image_resize)
        )

        # find the pixel resolution [h x w] from image as received
        height = cv_image.shape[0]
        width = cv_image.shape[1]
        image_size = height * width  # No. of pixels OR Total Area

        if config_.flip_image:  # flip the image if camera is mounted upside-down
            cv_image = cv2.flip(cv_image, 0)  # Upside down
            cv_image = cv2.flip(cv_image, 1)  # Side to side
        cam_image = cv_image  # Image to be used for Yolo detection - made global for use outside
    return


# ----------------------------------------------------------------


# Listener for yolo detection calls, routes them accordingly
def yolo_look_for_object_callback(String):
    global yolo_called
    yolo_called = True
    # When called
    look_for = String.data
    if look_for in {
        "stop sign",
        "stop-sign",
        "stop",
        "sign",
        "signs",
        "stop signs",
        "stop-signs",
        "tire",
        "tires",
        "pothole",
        "potholes",
        "pedestrian",
        "pedestrians",
        "person",
        "persons",
    }:
        detect_object()
    # if look_for in {"tire", "tires"}:
    #     detect_tire()
    # if look_for in {"pedestrian", "pedestrians", "person", "persons"}:
    #     detect_person()
    # ^^^ This allows to efficiently run these detections only when needed

    # Once detection is complete, call has ended
    yolo_called = False


# ----------------------------------------------------------------


# Make a local global copy of the parameter configuration
def dyn_rcfg_cb(config, level):
    global config_
    config_ = config
    return config


# ----------------------------------------------------------------


# Resize cv_image to desired size with black letterbox
# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def resize_image(img, size=(display_size, display_size)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)
    dif = h if h > w else w
    interpolation = (
        cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC
    )
    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2
    if len(img.shape) == 2:  # Grayscale images
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w] = img[:h, :w]
    else:  # 3-channel color images
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w, :] = img[:h, :w, :]
    return cv2.resize(mask, size, interpolation)


# ----------------------------------------------------------------


# Performs Inference and outputs the results
def infer_image_using(
    _path,  # Path to the .pt file
    _source,  # Image source
    _classes=None,  # List of classes to infer
    _agnostic_nms=True,  # Avoids multiple detections of the same object
    _device="0",  # Use GPU with cuda - device index "0"
    _stream=False,  # Used for video source, doesn't seem to help with images
    _verbose=False,  # Output results to terminal
    _conf=0.5,  # confidence threshold for detection
):
    from ultralytics import YOLO  # YOLOv8 object class

    model = YOLO(_path)  # Load YOLO model weights from file

    # Perform inference and output a Generator/Results object
    if _classes is None:  # Predict all classes
        results = model(
            source=_source,
            agnostic_nms=_agnostic_nms,
            device=_device,
            stream=_stream,
            verbose=_verbose,
            conf=_conf,
        )
    else:  # Check for specified classes
        results = model(
            source=_source,
            classes=_classes,
            agnostic_nms=_agnostic_nms,
            device=_device,
            stream=_stream,
            verbose=_verbose,
            conf=_conf,
        )

    return results  # Return direct output from the inference


# ----------------------------------------------------------------


# Take yolo v8 results object and analyze them for classes and bounding boxes
def analyze_results(results, classes, image_size_in_sq_pixels=409600):
    # Outputs initialization:
    global cam_image
    person_box = None
    sign_box = None
    detected = []
    biggest_bounding_boxes = []
    stop_sign_detected = 0
    tire_detected = 0
    pothole_detected = 0
    person_detected = 0
    stop_sign_biggest_bounding_box = 0
    tire_biggest_bounding_box = 0
    pothole_biggest_bounding_box = 0
    person_biggest_bounding_box = 0
    results_image = resize_image(results[0].plot())  # resize image for debug display

    # Check the results for the specified classes
    for result in results:
        # Copy boxes to CPU, then convert to numpy array
        boxes = result.boxes.cpu().numpy()  # Boxes object for bbox outputs
        labels = result.names  # Direct class labels access

        for idx, box in enumerate(boxes):  # Iterate bounding boxes and find the largest
            label = labels[int(box.cls)]  # Get the class label for this box
            if (
                label == "stop-sign"
            ):  # Check if label matches the class we are looking for
                stop_sign_detected += 1  # Counter for individual detections
                # Find the width and height of the bounding box
                sign_box_width = box.xywh[0][2]
                sign_box_height = box.xywh[0][3]
                sign_area = 100 * (
                    (sign_box_width * sign_box_height) / image_size_in_sq_pixels
                )  # Percent Area
                if (
                    sign_area > stop_sign_biggest_bounding_box
                ):  # Store the largest bounding box
                    stop_sign_biggest_bounding_box = sign_area
                    sign_box = box  # Store the sign box for OCR
            if label == "tire":  # Check if label matches the class we are looking for
                tire_detected += 1  # Counter for individual detections
                # Find the width and height of the bounding box
                tire_box_width = box.xywh[0][2]
                tire_box_height = box.xywh[0][3]
                tire_area = 100 * (
                    (tire_box_width * tire_box_height) / image_size_in_sq_pixels
                )  # Percent Area
                if (
                    tire_area > tire_biggest_bounding_box
                ):  # Store the largest bounding box
                    tire_biggest_bounding_box = tire_area
            if (
                label == "pothole"
            ):  # Check if label matches the class we are looking for
                pothole_detected += 1  # Counter for individual detections
                # Find the width and height of the bounding box
                pothole_box_width = box.xywh[0][2]
                pothole_box_height = box.xywh[0][3]
                pothole_area = 100 * (
                    (pothole_box_width * pothole_box_height) / image_size_in_sq_pixels
                )  # Percent Area
                if (
                    pothole_area > pothole_biggest_bounding_box
                ):  # Store the largest bounding box
                    pothole_biggest_bounding_box = pothole_area
            if label == "person":  # Check if label matches the class we are looking for
                person_detected += 1  # Counter for individual detections
                # Find the width and height of the bounding box
                person_box_width = box.xywh[0][2]
                person_box_height = box.xywh[0][3]
                person_area = 100 * (
                    (person_box_width * person_box_height) / image_size_in_sq_pixels
                )  # Percent Area
                if (
                    person_area > person_biggest_bounding_box
                ):  # Store the largest bounding box
                    person_biggest_bounding_box = person_area
                    person_box = box  # Store person boudning box for vest detection

    # Store detected object info and bounding box area in arrays
    detected.append(stop_sign_detected)
    detected.append(tire_detected)
    detected.append(pothole_detected)
    detected.append(person_detected)
    biggest_bounding_boxes.append(int(stop_sign_biggest_bounding_box * 100))
    biggest_bounding_boxes.append(int(tire_biggest_bounding_box * 100))
    biggest_bounding_boxes.append(int(pothole_biggest_bounding_box * 100))
    biggest_bounding_boxes.append(int(person_biggest_bounding_box * 100))

    # Return the count of detections, max bounding box size and plotted image
    return (detected, biggest_bounding_boxes, person_box, sign_box, results_image)


# ----------------------------------------------------------------


# Release Tensors on GPU memory associated with a particular model instance
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    # Needs to be run twice to release most tensors used by PyTorch

    # However, it does not release it for use outside of this process
    # That won't happen until this process dies (PyTorch design flaw)


# ----------------------------------------------------------------

# ----------------------------------------------------------------

# Runs unified model object detection


def detect_object():
    global cam_image
    global real_stop_sign_detected
    global ocr_reader
    # Detect objects
    (
        objects_detected,
        objects_biggest_bounding_boxes,
        person_box,
        sign_box,
        results_image,
    ) = analyze_results(
        infer_image_using(_path=model_u_model_path, _source=cam_image),
        classes={"stop-sign", "tire", "pothole", "person"},
        image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
    )

    stop_sign_msg = UInt8()
    stop_sign_size_msg = UInt32()
    tire_msg = UInt8()
    tire_size_msg = UInt32()
    pothole_msg = UInt8()
    pothole_size_msg = UInt32()
    person_msg = UInt8()
    person_size_msg = UInt32()

    # objects_detected array with num of detected objects:
    # index 0 = stop_sign_detected
    # index 1 = tire_detected
    # index 2 = pothole_detected
    # index 3 = person_detected

    # objects_biggest_bounding_boxes array with biggest bounding box area for each class:
    # index 0 = stop sign biggest box
    # index 1 = tire biggest box
    # index 2 = pothole biggest box
    # index 3 = person biggest box

    # Publish stop sign data
    if objects_detected[0] > 0:  # stop signs detected > 0
        # Get the xywh info from the sign bounding box
        sign_box_x_pos = sign_box.xywh[0][0]
        sign_box_y_pos = sign_box.xywh[0][1]
        sign_box_width = sign_box.xywh[0][2]
        sign_box_height = sign_box.xywh[0][3]

        # Crop the image to the size of the bounding box
        sign_box_left = int(sign_box_x_pos - 0.5 * sign_box_width)
        sign_box_right = int(sign_box_x_pos + 0.5 * sign_box_width)
        sign_box_top = int(sign_box_y_pos - 0.5 * sign_box_height)
        sign_box_bottom = int(sign_box_y_pos + 0.5 * sign_box_height)

        # Extract the bounding box
        sign_extracted = cam_image[
            sign_box_top:sign_box_bottom, sign_box_left:sign_box_right
        ]

        sign_debug = bridge.cv2_to_imgmsg(sign_extracted, "bgr8")

        if config_.enable_sign_box:
            sign_box_pub.publish(sign_debug)

        

        result = ocr_reader.readtext(
            sign_extracted, detail=0
        )  # extract the text from the image

        # Iterates through the results and checks if STOP is in the text
        for text in result:
            if "STOP" in text.upper():
                real_stop_sign_detected = True
                break
            else:
                real_stop_sign_detected = False

        if real_stop_sign_detected:  # if STOP is detected
            stop_sign_msg.data = 1
            sign_detect_pub.publish(stop_sign_msg)
            stop_sign_size_msg.data = objects_biggest_bounding_boxes[0]
            sign_size_pub.publish(stop_sign_size_msg)
        else:
            stop_sign_msg.data = 0
            sign_detect_pub.publish(stop_sign_msg)
            stop_sign_size_msg.data = 0
            sign_size_pub.publish(stop_sign_size_msg)
    else:
        stop_sign_msg.data = 0
        sign_detect_pub.publish(stop_sign_msg)
        stop_sign_size_msg.data = 0
        sign_size_pub.publish(stop_sign_size_msg)

    # Publish tire data
    if objects_detected[1] > 0:  # tires detected > 0
        tire_msg.data = objects_detected[1]
        tire_detect_pub.publish(tire_msg)
        tire_size_msg.data = objects_biggest_bounding_boxes[1]
        tire_size_pub.publish(tire_size_msg)
    else:
        tire_msg.data = 0
        tire_detect_pub.publish(tire_msg)
        tire_size_msg.data = 0
        tire_size_pub.publish(tire_size_msg)

    # Publish pothole data
    if objects_detected[2] > 0:  # potholes detected > 0
        pothole_msg.data = objects_detected[2]
        pothole_detect_pub.publish(pothole_msg)
        pothole_size_msg.data = objects_biggest_bounding_boxes[2]
        pothole_size_pub.publish(pothole_size_msg)
    else:
        pothole_msg.data = 0
        pothole_detect_pub.publish(pothole_msg)
        pothole_size_msg.data = 0
        pothole_size_pub.publish(pothole_size_msg)

    # Publish person data
    if objects_detected[3] > 0:  # persons detected > 0
        person_msg.data = objects_detected[3]
        person_detect_pub.publish(person_msg)
        person_size_msg.data = objects_biggest_bounding_boxes[3]
        person_size_pub.publish(person_size_msg)
    else:
        person_msg.data = 0
        person_detect_pub.publish(person_msg)
        person_size_msg.data = 0
        person_size_pub.publish(person_size_msg)

    if config_.debug:
        # Show the detected images real-time for debug
        debug_img = bridge.cv2_to_imgmsg(results_image, "bgr8")
        object_debug_pub.publish(debug_img)

    if config_.enable_vest_mask and person_box is not None:  # Show the vest mask
        # Get the results image, make it black and paste the person bounding box with mask applied onto image
        hsv_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2HSV)

        # Define lower and upper HSV thresholds for bright orange
        lower_orange = np.array(
            [config_.vest_mask_l_hue, config_.vest_mask_l_sat, config_.vest_mask_l_lum]
        )
        upper_orange = np.array(
            [config_.vest_mask_h_hue, config_.vest_mask_h_sat, config_.vest_mask_h_lum]
        )

        mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

        # Get the xywh info from the person bounding box
        person_box_x_pos = person_box.xywh[0][0]
        person_box_y_pos = person_box.xywh[0][1]
        person_box_width = person_box.xywh[0][2]
        person_box_height = person_box.xywh[0][3]

        person_box_x_low = int(person_box_x_pos - 0.5 * person_box_width)
        person_box_x_high = int(person_box_x_pos + 0.5 * person_box_width)
        person_box_y_low = int(person_box_y_pos - 0.5 * person_box_height)
        person_box_y_high = int(person_box_y_pos + 0.5 * person_box_height)

        # Extract the bounding box from the mask image
        vest_mask_img_box = mask[
            person_box_y_low:person_box_y_high, person_box_x_low:person_box_x_high
        ]

        # Apply the black_mask to the mask image to make black_img
        black_mask = np.zeros_like(mask)
        black_img = cv2.bitwise_and(mask, black_mask)

        # Paste the vest_mask_img_box onto the black_img
        black_img[
            person_box_y_low:person_box_y_high, person_box_x_low:person_box_x_high
        ] = vest_mask_img_box

        # Convert new black image to imgmsg
        debug_img = bridge.cv2_to_imgmsg(black_img, "bgr8")
        vest_mask_pub.publish(debug_img)

    clear_gpu_memory()


# # Runs stop sign detection and watches for fake signs
# def detect_stop_sign():
#     # print("start")
#     # Detect stop signs
#     (detected, biggest_bounding_box, results_image) = analyze_results(
#         infer_image_using(_path=model_coco_path, _source=cam_image, _classes=11),
#         classes={"stop sign"},
#         image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
#     )  # {11: "stop sign"}
#     # Detect fake stop signs
#     (fake_detected, fake_biggest_bounding_box, fake_results_image) = analyze_results(
#         infer_image_using(_path=model_stop_path, _source=cam_image, _conf=0.3),
#         classes={'stop-sign-fake'},
#         image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
#     )  # {0: 'stop-sign', 1: 'stop-sign-fake', 2: 'stop-sign-obstructed', 3: 'stop-sign-vandalized'}
#     if fake_detected > 0:
#         detected = 0
#         print("!! fake sign detected !!")
#     sign_msg = UInt8()
#     size_msg = UInt32()
#     # print(biggest_bounding_box)
#     # detected > fake_detected cases:
#     # 0 - 0 = no sign
#     # 1 - 0 = good sign
#     # 1 - 1 = fake sign
#     # 2 - 1 = fake sign and a good sign
#     # 0 - 1 = fake sign
#     # 1 - 2 = two fake signs
#     if detected > fake_detected:  # good sign detected
#         sign_msg.data = detected - fake_detected  # No. of good signs detected
#         sign_detect_pub.publish(sign_msg)
#         size_msg.data = biggest_bounding_box  # Bug: could be box of fake sign when two detected
#         sign_size_pub.publish(size_msg)
#     else:  # fake sign detected or no sign detected
#         sign_msg.data = 0
#         sign_detect_pub.publish(sign_msg)
#         size_msg.data = 0
#         sign_size_pub.publish(size_msg)

#     if config_.debug:
#         # Show the detected images real-time for debug
#         vertically_stacked_img = np.concatenate((results_image, fake_results_image), axis=0)
#         vertically_stacked_img = bridge.cv2_to_imgmsg(vertically_stacked_img, "bgr8")
#         sign_debug_pub.publish(vertically_stacked_img)

#     clear_gpu_memory()


# # ----------------------------------------------------------------


# # Runs tire detection
# def detect_tire():
#     # Detect tires
#     (detected, biggest_bounding_box, results_image) = analyze_results(
#         infer_image_using(_path=model_tire_path, _source=cam_image),
#         classes={"tire"},
#         image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
#     )  # {0: "tire"} ?????
#     # Publish results
#     tire_detect_msg = UInt8()
#     tire_size_msg = UInt32()
#     tire_detect_msg.data = detected
#     tire_detect_pub.publish(tire_detect_msg)
#     tire_size_msg.data = biggest_bounding_box
#     tire_size_pub.publish(tire_size_msg)

#     if config_.debug:
#         # Show the detected images real-time for debug
#         debug_img = bridge.cv2_to_imgmsg(results_image, "bgr8")
#         tire_debug_pub.publish(debug_img)


# # ----------------------------------------------------------------


# # Runs person detection
# def detect_person():
#     # Detect persons
#     (detected, biggest_bounding_box, results_image) = analyze_results(
#         infer_image_using(_path=model_coco_path, _source=cam_image, _classes=0),
#         classes="person",
#         image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
#     )  # {0: "person"}
#     # Publish results
#     person_detect_msg = UInt8()
#     person_size_msg = UInt32()
#     person_detect_msg.data = detected
#     person_detect_pub.publish(person_detect_msg)
#     person_size_msg.data = biggest_bounding_box
#     person_size_pub.publish(person_size_msg)

#     if config_.debug:
#         # Show the detected images real-time for debug
#         debug_img = bridge.cv2_to_imgmsg(results_image, "bgr8")
#         person_debug_pub.publish(debug_img)


# ----------------------------------------------------------------


########################################################################
### Main loop:
if __name__ == "__main__":
    # ROS Node name
    rospy.init_node("route_yolov8_detector", anonymous=False)

    # Dynamic Reconfigure parameter server
    srv = Server(
        YoloDetectConfig, dyn_rcfg_cb
    )  # Using common cfg file for entire project

    # Load the YOLO models at startup
    model_coco_path = rospy.get_param("~model_coco_path_from_root")  # Load latest path
    model_stop_path = rospy.get_param("~model_stop_path_from_root")  # Load latest path
    model_tire_path = rospy.get_param("~model_tire_path_from_root")  # Load latest path
    model_u_model_path = rospy.get_param(
        "~model_u_model_path_from_root"
    )  # Load u_model path

    # Image input from topic - from launch file
    imgtopic = rospy.get_param("~imgtopic_name")
    rospy.Subscriber(imgtopic, Image, get_image_callback, queue_size=1)

    # Listener to identify when to process images using yolo models
    call_topic = rospy.get_param("~look_for_object_topic_name")
    rospy.Subscriber(call_topic, String, yolo_look_for_object_callback, queue_size=1)

    # >>> Topics and publishers to output detection results at
    # Stop Sign detection:
    sign_detect_topic = rospy.get_param("~stop_sign_detected_topic_name")
    sign_size_topic = rospy.get_param("~stop_sign_size_topic_name")
    # sign_debug_topic = rospy.get_param("~stop_sign_debug_topic_name")
    sign_detect_pub = rospy.Publisher(
        sign_detect_topic, UInt8, queue_size=1
    )  # No. of signs detected
    sign_size_pub = rospy.Publisher(
        sign_size_topic, UInt32, queue_size=1
    )  # Biggest sign as % area of image
    # sign_debug_pub = rospy.Publisher(sign_debug_topic, Image, queue_size=1)  # Anotated image debug output

    # person detection:
    person_detect_topic = rospy.get_param("~person_detected_topic_name")
    person_size_topic = rospy.get_param("~person_size_topic_name")
    # person_debug_topic = rospy.get_param("~person_debug_topic_name")
    person_detect_pub = rospy.Publisher(
        person_detect_topic, UInt8, queue_size=1
    )  # No. of persons detected
    person_size_pub = rospy.Publisher(
        person_size_topic, UInt32, queue_size=1
    )  # Biggest person as % area of image
    # person_debug_pub = rospy.Publisher(person_debug_topic, Image, queue_size=1)  # Anotated image debug output

    # Tire detection:
    tire_detect_topic = rospy.get_param("~tire_detected_topic_name")
    tire_size_topic = rospy.get_param("~tire_size_topic_name")
    # tire_debug_topic = rospy.get_param("~tire_debug_topic_name")
    tire_detect_pub = rospy.Publisher(
        tire_detect_topic, UInt8, queue_size=1
    )  # No. of tires detected
    tire_size_pub = rospy.Publisher(
        tire_size_topic, UInt32, queue_size=1
    )  # Biggest tire as % area of image
    # tire_debug_pub = rospy.Publisher(tire_debug_topic, Image, queue_size=1)  # Anotated image debug output

    # Pothole detection:
    pothole_detect_topic = rospy.get_param("~pothole_detected_topic_name")
    pothole_size_topic = rospy.get_param("~pothole_size_topic_name")
    # pothole_debug_topic = rospy.get_param("~pothole_debug_topic_name")
    pothole_detect_pub = rospy.Publisher(
        pothole_detect_topic, UInt8, queue_size=1
    )  # No. of tires detected
    pothole_size_pub = rospy.Publisher(
        pothole_size_topic, UInt32, queue_size=1
    )  # Biggest tire as % area of image
    # pothole_debug_pub = rospy.Publisher(pothole_debug_topic, Image, queue_size=1)  # Anotated image debug output

    # Object debug:
    object_debug_topic = rospy.get_param("~object_debug_topic_name")
    object_debug_pub = rospy.Publisher(
        object_debug_topic, Image, queue_size=1
    )  # Anotated image debug output

    # >>> Topics and publishers for person vest mask and sign box
    # Person bounding box with mask for vest extraction
    vest_mask_topic = rospy.get_param("~vest_mask_topic_name")
    vest_mask_pub = rospy.Publisher(vest_mask_topic, Image, queue_size=1)

    # Sign bounding box
    sign_box_topic = rospy.get_param("~sign_box_topic_name")
    sign_box_pub = rospy.Publisher(sign_box_topic, Image, queue_size=1)

    rospy.spin()  # Runs callbacks

    # # Start Looping
    # try:
    #     while not rospy.is_shutdown():
    #         rospy.spin()  # Runs callbacks
    # except rospy.ROSInterruptException:
    #     cv2.destroyAllWindows()  # Close all windows
