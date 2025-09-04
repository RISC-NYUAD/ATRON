#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import json
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from rcl_interfaces.msg import ParameterDescriptor
import numpy as np


class BBoxVisualizer(Node):
    def __init__(self):
        super().__init__('bbox_visualizer')
        
        # Declare parameters
        self.declare_parameter(
            'image_topic',
            '/rgb',
            ParameterDescriptor(description='Input image topic')
        )
        self.declare_parameter(
            'bbox_topic',
            '/bbox_2d_tight',
            ParameterDescriptor(description='Detection2DArray topic with bounding boxes')
        )
        self.declare_parameter(
            'semantic_topic',
            '/semantic_labels',
            ParameterDescriptor(description='Semantic labels topic with class names')
        )
        self.declare_parameter(
            'output_topic',
            '/rgb/annotated',
            ParameterDescriptor(description='Output image topic with drawn bounding boxes')
        )
        self.declare_parameter(
            'marker_topic',
            '/detection_markers',
            ParameterDescriptor(description='Topic to publish 3D markers for all detections')
        )
        
        # Camera intrinsic parameters
        self.declare_parameter(
            'cx',
            240.0,
            ParameterDescriptor(description='Camera principal point x-coordinate')
        )
        self.declare_parameter(
            'cy',
            240.0,
            ParameterDescriptor(description='Camera principal point y-coordinate')
        )
        self.declare_parameter(
            'fx',
            240.0,
            ParameterDescriptor(description='Camera focal length in x')
        )
        self.declare_parameter(
            'fy',
            240.0,
            ParameterDescriptor(description='Camera focal length in y')
        )
        self.declare_parameter(
            'fov',
            1.57079632,
            ParameterDescriptor(description='Camera field of view in radians')
        )
        self.declare_parameter(
            'height',
            1.507,
            ParameterDescriptor(description='Camera height above water in meters')
        )
        self.declare_parameter(
            'offset',
            0.5,
            ParameterDescriptor(description='Forward offset for projected points in meters')
        )
        self.declare_parameter(
            'marker_lifetime',
            1.0,
            ParameterDescriptor(description='Marker lifetime in seconds (0 for infinite)')
        )
        self.declare_parameter(
            'color_code',
            False,
            ParameterDescriptor(description='Enable direction-specific colors for bounding boxes and markers')
        )
        self.declare_parameter(
            'camera_direction',
            'front',
            ParameterDescriptor(description='Camera direction (front, back, left, right) for color coding')
        )
        
        # Get parameter values
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        bbox_topic = self.get_parameter('bbox_topic').get_parameter_value().string_value
        semantic_topic = self.get_parameter('semantic_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        
        # Get camera parameters
        self.cx = self.get_parameter('cx').get_parameter_value().double_value
        self.cy = self.get_parameter('cy').get_parameter_value().double_value
        self.fx = self.get_parameter('fx').get_parameter_value().double_value
        self.fy = self.get_parameter('fy').get_parameter_value().double_value
        self.fov = self.get_parameter('fov').get_parameter_value().double_value
        self.height = self.get_parameter('height').get_parameter_value().double_value
        self.offset = self.get_parameter('offset').get_parameter_value().double_value
        self.marker_lifetime = self.get_parameter('marker_lifetime').get_parameter_value().double_value
        self.color_code = self.get_parameter('color_code').get_parameter_value().bool_value
        self.camera_direction = self.get_parameter('camera_direction').get_parameter_value().string_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create publishers
        self.image_pub = self.create_publisher(Image, output_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        
        # Store latest semantic labels
        self.class_labels = {}
        
        # Subscribe to semantic labels
        self.semantic_sub = self.create_subscription(
            String,
            semantic_topic,
            self.semantic_callback,
            10
        )
        
        # Create synchronized subscribers
        self.image_sub = Subscriber(self, Image, image_topic)
        self.bbox_sub = Subscriber(self, Detection2DArray, bbox_topic)
        
        # Use approximate time synchronizer since semantic labels might not be perfectly synced
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.bbox_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)
        
        self.get_logger().info(f'BBox Visualizer started')
        self.get_logger().info(f'Subscribing to image: {image_topic}')
        self.get_logger().info(f'Subscribing to bboxes: {bbox_topic}')
        self.get_logger().info(f'Subscribing to semantic labels: {semantic_topic}')
        self.get_logger().info(f'Publishing annotated images to: {output_topic}')
        self.get_logger().info(f'Publishing 3D markers to: {marker_topic}')
        self.get_logger().info(f'Camera parameters: cx={self.cx}, cy={self.cy}, fx={self.fx}, fy={self.fy}')
        self.get_logger().info(f'Projection parameters: fov={self.fov:.3f} rad, height={self.height}m, offset={self.offset}m')
        
        # Color map for different classes
        self.colors = {}
        self.color_index = 0
        self.available_colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (128, 0, 0),     # Dark Red
            (0, 128, 0),     # Dark Green
            (0, 0, 128),     # Dark Blue
            (255, 128, 0),   # Orange
        ]
    
    def get_direction_colors(self, class_name):
        """Get direction-specific colors for bounding boxes when color_code is enabled
        Colors match those used in orienteering.py for consistency"""
        if not self.color_code:
            # Use default colors when color coding is disabled
            if class_name == 'buoy':
                return (0, 165, 255)  # Orange (BGR format for OpenCV)
            else:  # debris
                return (0, 0, 255)    # Red (BGR format for OpenCV)
        
        # Direction-specific colors when color_code is enabled
        # Convert from RGB (orienteering.py) to BGR (OpenCV) format
        direction_colors = {
            'front': {
                'buoy': (0, 204, 0),      # Green (0.0, 0.8, 0.0) -> (0, 204, 0)
                'debris': (102, 255, 102) # Light green (0.4, 1.0, 0.4) -> (102, 255, 102)
            },
            'back': {
                'buoy': (255, 102, 0),    # Blue (0.0, 0.4, 1.0) -> (255, 102, 0)
                'debris': (255, 178, 102) # Light blue (0.4, 0.7, 1.0) -> (255, 178, 102)
            },
            'left': {
                'buoy': (0, 0, 255),      # Red (1.0, 0.0, 0.0) -> (0, 0, 255)
                'debris': (153, 102, 255) # Light red/pink (1.0, 0.4, 0.6) -> (153, 102, 255)
            },
            'right': {
                'buoy': (0, 204, 255),    # Yellow/orange (1.0, 0.8, 0.0) -> (0, 204, 255)
                'debris': (102, 255, 255) # Light yellow (1.0, 1.0, 0.4) -> (102, 255, 255)
            }
        }
        
        # Get colors for this camera direction
        colors = direction_colors.get(self.camera_direction, {
            'buoy': (0, 165, 255),  # Orange fallback
            'debris': (0, 0, 255)   # Red fallback
        })
        
        return colors.get(class_name, (0, 165, 255))  # Default to orange

    def get_color_for_class(self, class_name):
        """Get a consistent color for each class"""
        if not self.color_code:
            # Use default colors when color coding is disabled
            if class_name == 'buoy':
                return (0, 165, 255)  # Orange (BGR format for OpenCV)
            else:  # debris
                return (0, 0, 255)    # Red (BGR format for OpenCV)
        
        if class_name not in self.colors:
            self.colors[class_name] = self.available_colors[self.color_index % len(self.available_colors)]
            self.color_index += 1
        return self.colors[class_name]
    
    def semantic_callback(self, msg):
        """Process semantic labels message"""
        try:
            labels_data = json.loads(msg.data)
            # Extract class labels, ignoring timestamp
            self.class_labels = {}
            for key, value in labels_data.items():
                if key != 'time_stamp' and isinstance(value, dict) and 'class' in value:
                    self.class_labels[key] = value['class']
            self.get_logger().debug(f'Updated class labels: {self.class_labels}')
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse semantic labels: {e}')
    
    def callback(self, image_msg, bbox_array_msg):
        """Process synchronized image and detection messages"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # Create marker array for all detections
            marker_array = MarkerArray()
            
            # Handle case with no detections
            if not bbox_array_msg.detections:
                # Publish empty marker array to clear any existing markers
                empty_marker_array = MarkerArray()
                self.marker_pub.publish(empty_marker_array)
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                return
            
            # Vectorized 3D projection - collect all bbox data into arrays
            n_detections = len(bbox_array_msg.detections)
            center_x = np.zeros(n_detections)
            center_y = np.zeros(n_detections)
            size_y = np.zeros(n_detections)
            
            for i, detection in enumerate(bbox_array_msg.detections):
                center_x[i] = detection.bbox.center.position.x
                center_y[i] = detection.bbox.center.position.y
                size_y[i] = detection.bbox.size_y
            
            # Compute bottom_y for all detections
            bottom_y = center_y + size_y / 2
            
            # Vectorized projection calculations
            pixels = np.column_stack([center_x, bottom_y])
            pixels_centered = pixels - np.array([self.cx, self.cy])
            angles = (self.fov/2) * pixels_centered / np.array([self.fx, self.fy])
            
            # Compute z and x for all points at once
            z_values = self.height / np.tan(angles[:, 1]) + self.offset
            x_values = z_values * np.tan(angles[:, 0])
            
            # Process each detection in the array
            for i, detection in enumerate(bbox_array_msg.detections):
                # Get bounding box
                bbox = detection.bbox
                
                # Calculate corner points from center and size
                bbox_center_x = int(center_x[i])
                bbox_center_y = int(center_y[i])
                half_width = int(bbox.size_x / 2)
                half_height = int(bbox.size_y / 2)
                
                # Top-left and bottom-right corners
                x1 = bbox_center_x - half_width
                y1 = bbox_center_y - half_height
                x2 = bbox_center_x + half_width
                y2 = bbox_center_y + half_height
                
                # Get best classification result
                if detection.results:
                    best_result = max(detection.results, key=lambda r: r.hypothesis.score)
                    class_id = best_result.hypothesis.class_id
                    score = best_result.hypothesis.score
                    
                    # Get class name from semantic labels
                    class_name = self.class_labels.get(class_id, f"class_{class_id}")
                    
                    # Get color for this class (use direction-specific colors if enabled)
                    if self.color_code:
                        color = self.get_direction_colors(class_name)
                    else:
                        color = self.get_color_for_class(class_name)
                    
                    # Draw bounding box
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label
                    # label = f"{class_name}: {score:.2f}"
                    label = f"{class_name}"
                    
                    # Draw label background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw filled rectangle for label background
                    cv2.rectangle(cv_image, 
                                (x1, y1 - label_height - 4),
                                (x1 + label_width + 4, y1),
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(cv_image, label,
                              (x1 + 2, y1 - 2),
                              font, font_scale,
                              (255, 255, 255),  # White text
                              thickness, cv2.LINE_AA)
                    
                    # Create 3D marker for detection
                    marker = Marker()
                    marker.header = image_msg.header
                    marker.id = i
                    marker.ns = "detections"
                    
                    # Use pre-computed 3D positions
                    marker.pose.position.x = x_values[i]
                    marker.pose.position.y = self.height
                    marker.pose.position.z = z_values[i]
                    
                    # Get color from bounding box (BGR) and convert to RGB
                    if self.color_code:
                        bbox_color = self.get_direction_colors(class_name)
                    else:
                        bbox_color = self.get_color_for_class(class_name)
                    marker.color.r = bbox_color[2] / 255.0  # BGR to RGB
                    marker.color.g = bbox_color[1] / 255.0
                    marker.color.b = bbox_color[0] / 255.0
                    marker.color.a = 1.0
                    
                    # Set marker type and scale based on class
                    if class_name == 'buoy':  # Buoy
                        marker.type = Marker.CYLINDER
                        marker.scale.x = 1.0  # diameter = 2 * radius = 2 * 0.5 = 1.0
                        marker.scale.y = 1.0  # diameter
                        marker.scale.z = self.height  # height
                        
                        # Adjust position so cylinder extends upward from water surface
                        marker.pose.position.y = self.height / 2.0  # Center of cylinder at half height
                        
                        # Rotate cylinder to align with y-axis pointing up (-90 degrees around x-axis)
                        marker.pose.orientation.x = -0.707
                        marker.pose.orientation.y = 0.0
                        marker.pose.orientation.z = 0.0
                        marker.pose.orientation.w = 0.707
                    else:  # Debris (can, cup, bottle)
                        marker.type = Marker.SPHERE
                        marker.scale.x = 0.3  # diameter = 2 * radius = 2 * 0.05 = 0.1
                        marker.scale.y = 0.3
                        marker.scale.z = 0.3
                        
                        # No rotation needed for spheres
                        marker.pose.orientation.x = 0.0
                        marker.pose.orientation.y = 0.0
                        marker.pose.orientation.z = 0.0
                        marker.pose.orientation.w = 1.0
                    
                    # Set marker lifetime
                    if self.marker_lifetime > 0:
                        marker.lifetime = rclpy.duration.Duration(seconds=self.marker_lifetime).to_msg()
                    
                    marker.action = Marker.ADD
                    marker_array.markers.append(marker)
                    
                    self.get_logger().debug(f'Created {class_name} marker at ({x_values[i]:.3f}, {self.height:.3f}, {z_values[i]:.3f})')
            
            # Publish marker array
            if marker_array.markers:
                self.marker_pub.publish(marker_array)
                self.get_logger().debug(f'Published {len(marker_array.markers)} markers')
            
            # Convert back to ROS image and publish
            output_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            output_msg.header = image_msg.header
            self.image_pub.publish(output_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = BBoxVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# Original non-vectorized implementation for reference:
"""
# In the callback loop for each detection:

# Use bottom of bbox for y-coordinate projection
bottom_y = center_y + half_height

# Apply the projection formula
theta_y = (self.fov/2) * (bottom_y - self.cy) / self.fy
z = self.height / np.tan(theta_y) + self.offset
theta_x = (self.fov/2) * (center_x - self.cx) / self.fx
x = z * np.tan(theta_x)

# Set marker position
marker.pose.position.x = x
marker.pose.position.y = self.height
marker.pose.position.z = z
"""
