#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image

# from me416_lab
import image_processing as ip

# convert OpenCV to Image using cv_bridge
bridge = CvBridge()

global pub1, pub2


def callback(msg):
    global pub1, pub2
    # convert Image msg to OpenCV image
    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    # resize image
    # img = cv2.resize(img,(320,240))
    # img = img[1:10,:,:]
    # run img_classify to segment test track line from the background
    lb, ub = ip.classifier_parameters()
    img_segmented = cv2.inRange(img, lb, ub)
    # run img_centroid_horizontal to compute centroid
    x_centroid = ip.image_centroid_horizontal(img_segmented)
    # run img_line_vertical to add the line on the segmented image
    img_segmented_line = ip.image_line_vertical(
        ip.image_one_to_three_channels(img_segmented), x_centroid)
    img_msg = bridge.cv2_to_imgmsg(img_segmented_line, "bgr8")
    # publish segmented image to topic /image/segmented
    pub1.publish(img_msg)
    x_centroid = float(x_centroid)  # PointStamped.point.x is a float
    t = rospy.Time.now()
    point_msg = PointStamped()
    point_msg.point.x = x_centroid  # update x field of PointStamped
    point_msg.header.stamp = t  # add current time to header.stamp
    pub2.publish(point_msg)  # publish to /image/centroid


def main():
    global pub1, pub2, msg, img_processed
    rospy.init_node('main')
    rospy.Subscriber('/raspicam_node/image',
                     Image,
                     callback=callback,
                     queue_size=1,
                     buff_size=2**18)
    pub1 = rospy.Publisher('/image/segmented', Image, queue_size=10)
    pub2 = rospy.Publisher('/image/centroid', PointStamped, queue_size=10)
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
