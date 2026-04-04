#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Video writer variables
        self.video_writer = None
        self.is_recording = False
        self.frame_count = 0
        self.fps = 30.0  # Default FPS, will be updated from actual frame rate
        self.frame_width = None
        self.frame_height = None
        
        # Create output directory
        self.output_dir = os.path.expanduser('~/facade_pose_videos')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_filename = os.path.join(self.output_dir, f'combined_video_{timestamp}.mp4')
        
        # Subscribe to the combined_image topic
        self.subscription = self.create_subscription(
            Image,
            'combined_image',
            self.image_callback,
            10
        )
        
        self.get_logger().info(f'Image saver node started. Saving to: {self.output_filename}')
        self.get_logger().info('Waiting for images on /combined_image topic...')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Get image dimensions
            height, width = cv_image.shape[:2]
            
            # Initialize video writer on first frame
            if not self.is_recording:
                self.frame_width = width
                self.frame_height = height
                
                # Get actual frame rate from message header if available
                if msg.header.stamp.sec > 0:
                    # Could calculate FPS from timestamps, but for now use default
                    pass
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
                self.video_writer = cv2.VideoWriter(
                    self.output_filename,
                    fourcc,
                    self.fps,
                    (self.frame_width, self.frame_height)
                )
                
                if self.video_writer.isOpened():
                    self.is_recording = True
                    self.get_logger().info(f'Started recording video: {self.output_filename}')
                    self.get_logger().info(f'Resolution: {width}x{height}, FPS: {self.fps}')
                else:
                    self.get_logger().error('Failed to open video writer!')
                    return
            
            # Check if resolution changed
            if width != self.frame_width or height != self.frame_height:
                self.get_logger().warn(f'Resolution changed from {self.frame_width}x{self.frame_height} to {width}x{height}')
                self.frame_width = width
                self.frame_height = height
                # Reinitialize video writer with new resolution
                self.video_writer.release()
                self.video_writer = cv2.VideoWriter(
                    self.output_filename,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (self.frame_width, self.frame_height)
                )
            
            # Write frame to video
            if self.is_recording and self.video_writer.isOpened():
                self.video_writer.write(cv_image)
                self.frame_count += 1
                
                # Log every 100 frames
                if self.frame_count % 100 == 0:
                    self.get_logger().info(f'Recorded {self.frame_count} frames')
                    
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def __del__(self):
        # Clean up video writer when node is destroyed
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f'Video saved: {self.output_filename}')
            self.get_logger().info(f'Total frames recorded: {self.frame_count}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()