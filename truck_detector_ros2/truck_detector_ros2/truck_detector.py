# AKAMAV
# project: imav2023
# author: Junbo Li
# Version: 2023.08.16


import rclpy
import message_filters
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from truck_detector_ros2.truck_detector_functions import *


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
node_name = "truck_detection_node"


class truck_detection_node(Node):
    def __init__(self):
        super().__init__(node_name)
        self.declare_parameter('cam_topic', rclpy.Parameter.Type.STRING)
        self.declare_parameter('cam_pos_topic', rclpy.Parameter.Type.STRING)
        self.declare_parameter('allow_delay', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('engine', rclpy.Parameter.Type.STRING)
        self.declare_parameter('device', rclpy.Parameter.Type.STRING)
        self.declare_parameter('scale_factor', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('update_frequency', rclpy.Parameter.Type.INTEGER)

        self.bridge = CvBridge()
        self.truck_posestamped = PoseStamped()
        # Get parameters(descriptions see above) 
        self.cam_topic        = self.get_parameter('cam_topic').get_parameter_value().string_value
        self.cam_pos_topic    = self.get_parameter('cam_pos_topic').get_parameter_value().string_value
        self.allow_delay      = self.get_parameter('allow_delay').get_parameter_value().bool_value
        self.engine           = self.get_parameter('engine').get_parameter_value().string_value
        self.device           = self.get_parameter('device').get_parameter_value().string_value
        self.scale_factor     = self.get_parameter('scale_factor').get_parameter_value().double_value
        self.update_frequency = self.get_parameter('update_frequency').get_parameter_value().integer_value
        # ROS node(Publisher)
        self.truck_pos_pub = self.create_publisher(PoseStamped, 'truck/posestamped', 10)
        # Initialize inference engine (TensorRT)
        self.bindings, self.binding_addrs, self.exec_context = Infer_init(self.engine, self.device)
        # Node cycle rate (in Hz).
        self.loop_rate = self.create_rate(self.update_frequency)
        # Subscribers
        t1 = message_filters.Subscriber(self, Image, self.cam_topic)  
        t2 = message_filters.Subscriber(self, PoseStamped, self.cam_pos_topic) 
        if not self.allow_delay:
            ts = message_filters.TimeSynchronizer([t1, t2], 10)
        else:
            ts = message_filters.ApproximateTimeSynchronizer([t1, t2], 10, 1)
        ts.registerCallback(self.truck_callback)    

    def truck_callback(self, ros_img, cam_pos):
        cv_image = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
        img, ratio, dwdh = img_preprocess(cv_image, self.device)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.exec_context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        scores = self.bindings['det_scores'].data
        classes = self.bindings['det_classes'].data
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
            self.truck_posestamped.header.stamp = self.get_clock().now().to_msg()
            self.truck_posestamped.pose = pose2msg_converter(truck_pos, cam_pos, cv_image.shape[0], cv_image.shape[1], self.scale_factor)
            self.truck_pos_pub.publish(self.truck_posestamped)
        visualizer(cv_image, truck_pos)
        self.loop_rate.sleep()

def main():
    rclpy.init()

    truck_detector = truck_detection_node()
      
    rclpy.spin(truck_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    truck_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()