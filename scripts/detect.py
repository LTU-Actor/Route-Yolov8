#!/usr/bin/env python3
########################################################################
import rospy  # ROS API
from sensor_msgs.msg import Image  # Image message
from std_msgs.msg import UInt8, UInt32, String  # UInt8(used as bool) and UInt32(used as +ve number) message
import ultralytics  # Yolo v8
import numpy as np  # numpy
import torch
import gc  # Garbage collection
import time
import cv2  # OpenCV version 2.x (ROS limitation)
from cv_bridge import CvBridge, CvBridgeError  # ROS to/from OpenCV API
from dynamic_reconfigure.server import Server  # ROS Parameter server for debug
from ltu_actor_route_yolov8_detector.cfg import YoloDetectConfig  # packageName.cfg

########################################################################
### Global Variables:
global config_  # Dynamic reconfiguration holder
global bridge  # ROS-CV bridge
bridge = CvBridge()

# >>> INITIALIZE MODEL PATHS HERE: <<<
global model_stop_path  # Get yolov8 stop sign detection model's path
global model_coco_path  # Get yolov8 COCO trained model's path
global model_tire_path  # Get yolov8 tire detection model's path

global display_size
display_size = 640  # pixel resolution used for debug outputs

global image_size  # pixel resolution used for inference
global cam_image  # image used for detection

global yolo_called
yolo_called = False  # True if yolo detection is called for from topic

########################################################################
### Functions:


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
        cv_image = resize_image(cv_image, size=(config_.image_resize, config_.image_resize))

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
    if look_for in {"stop sign", "stop-sign", "stop", "sign", "signs", "stop signs", "stop-signs"}:
        detect_stop_sign()
    if look_for in {"tire", "tires"}:
        detect_tire()
    if look_for in {"pedestrian", "pedestrians", "person", "persons"}:
        detect_person()
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
    interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC
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
    detected = 0
    biggest_bounding_box = 0
    results_image = resize_image(results[0].plot())  # resize image for debug display

    # Check the results for the specified classes
    for result in results:
        # Copy boxes to CPU, then convert to numpy array
        boxes = result.boxes.cpu().numpy()  # Boxes object for bbox outputs
        labels = result.names  # Direct class labels access

        for idx, box in enumerate(boxes):  # Iterate bounding boxes and find the largest
            label = labels[int(box.cls)]  # Get the class label for this box
            if label in classes:  # Check if label matches the class we are looking for
                detected += 1  # Counter for individual detections
                # Find the width and height of the bounding box
                box_width = box.xywh[0][2]
                box_height = box.xywh[0][3]
                area = 100 * ((box_width * box_height) / image_size_in_sq_pixels)  # Percent Area
                if area > biggest_bounding_box:  # Store the largest bounding box
                    biggest_bounding_box = area

    # Return the count of detections, max bounding box size and plotted image
    return (detected, int(biggest_bounding_box * 100), results_image)


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


# Runs stop sign detection and watches for fake signs
def detect_stop_sign():
    # print("start")
    # Detect stop signs
    (detected, biggest_bounding_box, results_image) = analyze_results(
        infer_image_using(_path=model_coco_path, _source=cam_image, _classes=11),
        classes={"stop sign"},
        image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
    )  # {11: "stop sign"}
    # Detect fake stop signs
    (fake_detected, fake_biggest_bounding_box, fake_results_image) = analyze_results(
        infer_image_using(_path=model_stop_path, _source=cam_image, _conf=0.3),
        classes={'stop-sign-fake'},
        image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
    )  # {0: 'stop-sign', 1: 'stop-sign-fake', 2: 'stop-sign-obstructed', 3: 'stop-sign-vandalized'}
    if fake_detected > 0:
        detected = 0
        print("!! fake sign detected !!")
    sign_msg = UInt8()
    size_msg = UInt32()
    # print(biggest_bounding_box)
    # detected > fake_detected cases:
    # 0 - 0 = no sign
    # 1 - 0 = good sign
    # 1 - 1 = fake sign
    # 2 - 1 = fake sign and a good sign
    # 0 - 1 = fake sign
    # 1 - 2 = two fake signs
    if detected > fake_detected:  # good sign detected
        sign_msg.data = detected - fake_detected  # No. of good signs detected
        sign_detect_pub.publish(sign_msg)
        size_msg.data = biggest_bounding_box  # Bug: could be box of fake sign when two detected
        sign_size_pub.publish(size_msg)
    else:  # fake sign detected or no sign detected
        sign_msg.data = 0
        sign_detect_pub.publish(sign_msg)
        size_msg.data = 0
        sign_size_pub.publish(size_msg)

    if config_.debug:
        # Show the detected images real-time for debug
        vertically_stacked_img = np.concatenate((results_image, fake_results_image), axis=0)
        vertically_stacked_img = bridge.cv2_to_imgmsg(vertically_stacked_img, "bgr8")
        sign_debug_pub.publish(vertically_stacked_img)

    clear_gpu_memory()


# ----------------------------------------------------------------


# Runs tire detection
def detect_tire():
    # Detect tires
    (detected, biggest_bounding_box, results_image) = analyze_results(
        infer_image_using(_path=model_tire_path, _source=cam_image),
        classes={"tire"},
        image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
    )  # {0: "tire"} ?????
    # Publish results
    tire_detect_msg = UInt8()
    tire_size_msg = UInt32()
    tire_detect_msg.data = detected
    tire_detect_pub.publish(tire_detect_msg)
    tire_size_msg.data = biggest_bounding_box
    tire_size_pub.publish(tire_size_msg)

    if config_.debug:
        # Show the detected images real-time for debug
        debug_img = bridge.cv2_to_imgmsg(results_image, "bgr8")
        tire_debug_pub.publish(debug_img)


# ----------------------------------------------------------------


# Runs person detection
def detect_person():
    # Detect persons
    (detected, biggest_bounding_box, results_image) = analyze_results(
        infer_image_using(_path=model_coco_path, _source=cam_image, _classes=0),
        classes="person",
        image_size_in_sq_pixels=(cam_image.shape[0] * cam_image.shape[1]),
    )  # {0: "person"}
    # Publish results
    person_detect_msg = UInt8()
    person_size_msg = UInt32()
    person_detect_msg.data = detected
    person_detect_pub.publish(person_detect_msg)
    person_size_msg.data = biggest_bounding_box
    person_size_pub.publish(person_size_msg)

    if config_.debug:
        # Show the detected images real-time for debug
        debug_img = bridge.cv2_to_imgmsg(results_image, "bgr8")
        person_debug_pub.publish(debug_img)


# ----------------------------------------------------------------


########################################################################
### Main loop:
if __name__ == "__main__":
    # ROS Node name
    rospy.init_node("route_yolov8_detector", anonymous=False)

    # Dynamic Reconfigure parameter server
    srv = Server(YoloDetectConfig, dyn_rcfg_cb)  # Using common cfg file for entire project

    # Load the YOLO models at startup
    model_coco_path = rospy.get_param("~model_coco_path_from_root")  # Load latest path
    model_stop_path = rospy.get_param("~model_stop_path_from_root")  # Load latest path
    model_tire_path = rospy.get_param("~model_tire_path_from_root")  # Load latest path

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
    sign_debug_topic = rospy.get_param("~stop_sign_debug_topic_name")
    sign_detect_pub = rospy.Publisher(sign_detect_topic, UInt8, queue_size=1)  # No. of signs detected
    sign_size_pub = rospy.Publisher(sign_size_topic, UInt32, queue_size=1)  # Biggest sign as % area of image
    sign_debug_pub = rospy.Publisher(sign_debug_topic, Image, queue_size=1)  # Anotated image debug output

    # person detection:
    person_detect_topic = rospy.get_param("~person_detected_topic_name")
    person_size_topic = rospy.get_param("~person_size_topic_name")
    person_debug_topic = rospy.get_param("~person_debug_topic_name")
    person_detect_pub = rospy.Publisher(person_detect_topic, UInt8, queue_size=1)  # No. of persons detected
    person_size_pub = rospy.Publisher(person_size_topic, UInt32, queue_size=1)  # Biggest person as % area of image
    person_debug_pub = rospy.Publisher(person_debug_topic, Image, queue_size=1)  # Anotated image debug output

    # Tire detection:
    tire_detect_topic = rospy.get_param("~tire_detected_topic_name")
    tire_size_topic = rospy.get_param("~tire_size_topic_name")
    tire_debug_topic = rospy.get_param("~tire_debug_topic_name")
    tire_detect_pub = rospy.Publisher(tire_detect_topic, UInt8, queue_size=1)  # No. of tires detected
    tire_size_pub = rospy.Publisher(tire_size_topic, UInt32, queue_size=1)  # Biggest tire as % area of image
    tire_debug_pub = rospy.Publisher(tire_debug_topic, Image, queue_size=1)  # Anotated image debug output

    rospy.spin()  # Runs callbacks

    # # Start Looping
    # try:
    #     while not rospy.is_shutdown():
    #         rospy.spin()  # Runs callbacks
    # except rospy.ROSInterruptException:
    #     cv2.destroyAllWindows()  # Close all windows
