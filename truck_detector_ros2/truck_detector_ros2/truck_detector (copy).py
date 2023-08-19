#!/usr/bin python
# AKAMAV
# project: imav2022
# author: Junbo Li
# Version: 2022.09.05


import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from truck_detector_functions import *


#---------- Parameters ----------
#   ~engine: model file (TensorRT engine)
#   ~device: GPU device number
#   ~scale_factor: scale factor for estimating the true coordinates of the truck
#   ~cam_topic: subscriber topic (camera frame)
#   ~cam_pos_topic: subscriber topic (camera pos)
#   ~update_frequency: update frequency in Hz
#   ~allow_delay: whether to allow Synchronization delay when subscribing to msgs (if True->faster)
#   NOTE: possible detection classes:
#       ['pedestrian',
#       'people',
#       'bicycle',
#       'car',
#       'van',
#       'truck',
#       'tricycle',
#       'awning-tricycle',
#       'bus',
#       'motor',
#       'others']
node_name = "truck_detector"
# parameters from launch file
cam_topic        = rospy.get_param(node_name + "/cam_topic")
cam_pos_topic    = rospy.get_param(node_name + "/cam_pos_topic")
allow_delay      = rospy.get_param(node_name + "/allow_delay")
engine           = rospy.get_param(node_name + "/engine")
device           = rospy.get_param(node_name + "/device")
scale_factor     = rospy.get_param(node_name + "/scale_factor")
update_frequency = rospy.get_param(node_name + "/update_frequency")


#---------- MAIN ----------
def truck_callback(ros_img, cam_pos):
    cv_image = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
    img, ratio, dwdh = img_preprocess(cv_image, device)
    binding_addrs['images'] = int(img.data_ptr())
    context.execute_v2(list(binding_addrs.values()))
    nums = bindings['num_dets'].data
    boxes = bindings['det_boxes'].data
    scores = bindings['det_scores'].data
    classes = bindings['det_classes'].data
    boxes = boxes[0,:nums[0][0]]
    scores = scores[0,:nums[0][0]]
    classes = classes[0,:nums[0][0]]
    # truck_pos in image plane(2D)
    truck_pos = truck_pos_estimator(ratio, dwdh, boxes, scores, classes)
    if truck_pos is not False:
        # truck_posestamped
            # reserved header information:
            #truck_posestamped.header.seq = ...
            #truck_posestamped.header.frame_id = ...
        truck_posestamped.header.stamp = rospy.Time.now()
        truck_posestamped.pose = pose2msg_converter(truck_pos, cam_pos, cv_image.shape[0], cv_image.shape[1], scale_factor)
        truck_pos_pub.publish(truck_posestamped)
    #visualizer(cv_image, truck_pos)
    loop_rate.sleep()


if __name__ == '__main__':
    try:
        bridge = CvBridge()
        truck_posestamped = PoseStamped()
        # ROS node(Publisher)
        rospy.init_node(node_name, anonymous = True)
        # Initialize inference engine (TensorRT)
        bindings, binding_addrs, context = Infer_init(engine, device)
        # Node cycle rate (in Hz).
        loop_rate = rospy.Rate(update_frequency)
        truck_pos_pub = rospy.Publisher("truck/posestamped", PoseStamped, queue_size = 10) 
        # Subscribers
        t1 = message_filters.Subscriber(cam_topic, Image)  
        t2 = message_filters.Subscriber(cam_pos_topic, PoseStamped) 
        if not allow_delay:
            ts = message_filters.TimeSynchronizer([t1, t2], 10)
        else:
            ts = message_filters.ApproximateTimeSynchronizer([t1, t2], 10, 1)
        ts.registerCallback(truck_callback)          
        rospy.spin()
    except rospy.ROSInterruptException:
        pass