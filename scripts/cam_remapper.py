#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo  # Adjust according to the message types you are using

class TopicRemapperNode:
    def __init__(self):
        # Publishers for the remapped topics
        self.depth_pub = rospy.Publisher('/usb_high_depth/image_raw', Image, queue_size=10)
        self.color_pub = rospy.Publisher('/usb_cam_high/image_raw', Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher('/usb_cam_high/camera_info', CameraInfo, queue_size=10)

        # Subscribers for the original topics
        rospy.Subscriber('/camera_2/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera_2/color/image_raw', Image, self.color_callback)
        rospy.Subscriber('/camera_2/color/camera_info', CameraInfo, self.camera_info_callback)

    def depth_callback(self, msg):
        # Simply republish the depth image message
        self.depth_pub.publish(msg)

    def color_callback(self, msg):
        # Simply republish the color image message
        self.color_pub.publish(msg)

    def camera_info_callback(self, msg):
        # Simply republish the camera info message
        self.camera_info_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('topic_remapper_node', anonymous=True)
    remapper = TopicRemapperNode()
    rospy.spin()
