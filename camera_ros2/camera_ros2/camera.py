import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class camera_node(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('width', '1920')
        self.declare_parameter('height', '1080')
        self.declare_parameter('fps', '30')

def main():
    rclpy.init()
    bridge = CvBridge()
    camera_pub_node = camera_node()
    #----------parameters----------
    width = camera_pub_node.get_parameter('width').get_parameter_value().string_value
    height = camera_pub_node.get_parameter('height').get_parameter_value().string_value
    framerate = camera_pub_node.get_parameter('fps').get_parameter_value().string_value

    pipeline = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={framerate}/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'.format(width=width, height=height, framerate=framerate)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    img_pub = camera_pub_node.create_publisher(Image, 'camera_image', 10)
    camera_pub_node.get_logger().info("Camera start!")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            image_message = bridge.cv2_to_imgmsg(frame, encoding='rgb8')
            image_message.header.stamp = camera_pub_node.get_clock().now().to_msg()
            img_pub.publish(image_message)


            #----------Debugging----------
            #cv2.imshow("camera", frame)
            #key = cv2.waitKey(1)
            #if key == 27:
            #    camera_pub_node.get_logger().info("quit!")
            #    break
        else:
            camera_pub_node.get_logger().info("No more frame!")
            break
    else:
        camera_pub_node.get_logger().info("VideoCapture error!")
    cap.release()
    cv2.destroyAllWindows()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_pub_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()