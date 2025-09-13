#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rcl_interfaces.msg import ParameterDescriptor
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class KalmanFilter3D:
    """Kalman filter for 3D object tracking"""
    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(6)
        
        # Measurement matrix (we observe x, y, z)
        self.H = np.eye(3, 6)
        
        # Process noise covariance
        self.Q = np.eye(6) * process_noise
        self.Q[3:, 3:] *= 10  # Higher noise for velocity
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise
        
        # State covariance matrix
        self.P = np.eye(6) * 100
        
        # Innovation covariance
        self.S = np.zeros((3, 3))
        
        # Kalman gain
        self.K = np.zeros((6, 3))
        
        self.initialized = False
        
    def initialize(self, x, y, z):
        """Initialize filter with first measurement"""
        self.state[:3] = [x, y, z]
        self.state[3:] = 0  # Zero initial velocity
        self.initialized = True
        
    def predict(self, dt):
        """Predict next state"""
        if not self.initialized:
            return
            
        # Update state transition matrix with time step
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, x, y, z):
        """Update state with new measurement"""
        if not self.initialized:
            self.initialize(x, y, z)
            return
            
        # Measurement
        z_meas = np.array([x, y, z])
        
        # Innovation
        y = z_meas - self.H @ self.state
        
        # Innovation covariance
        self.S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        
        # Update state
        self.state = self.state + self.K @ y
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - self.K @ self.H) @ self.P
        
    def get_position(self):
        """Get filtered position"""
        return self.state[:3]
        
    def get_velocity(self):
        """Get estimated velocity"""
        return self.state[3:]


class TrackedObject:
    """Represents a tracked object with Kalman filter"""
    def __init__(self, track_id, bbox, position_3d, class_id, confidence, process_noise=0.1, measurement_noise=1.0):
        self.id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.kalman = KalmanFilter3D(process_noise, measurement_noise)
        self.kalman.initialize(*position_3d)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.last_update_time = None
        
    def predict(self, current_time):
        """Predict next state"""
        if self.last_update_time is not None:
            dt = (current_time - self.last_update_time).nanoseconds / 1e9
            self.kalman.predict(dt)
        self.age += 1
        self.time_since_update += 1
        
    def update(self, bbox, position_3d, confidence, current_time):
        """Update with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.kalman.update(*position_3d)
        self.hits += 1
        self.time_since_update = 0
        self.last_update_time = current_time
        
    def get_state(self):
        """Get current state for marker publishing"""
        pos = self.kalman.get_position()
        return {
            'id': self.id,
            'position': pos,
            'velocity': self.kalman.get_velocity(),
            'bbox': self.bbox,
            'class_id': self.class_id,
            'confidence': self.confidence
        }


class ObjectTracker:
    """Multi-object tracker with data association"""
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3, 
                 process_noise=0.1, measurement_noise=1.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks = []
        self.next_id = 0
        
    def update(self, detections, current_time):
        """
        Update tracker with new detections
        detections: list of dicts with keys: bbox, position_3d, class_id, confidence
        """
        # Predict all tracks
        for track in self.tracks:
            track.predict(current_time)
            
        # Associate detections to tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            for d, det in enumerate(detections):
                for t, track in enumerate(self.tracks):
                    iou_matrix[d, t] = self._calculate_iou(det['bbox'], track.bbox)
                    
            # Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            # Update matched tracks
            matched_detections = set()
            matched_tracks = set()
            
            for d, t in zip(row_ind, col_ind):
                if iou_matrix[d, t] >= self.iou_threshold:
                    self.tracks[t].update(
                        detections[d]['bbox'],
                        detections[d]['position_3d'],
                        detections[d]['confidence'],
                        current_time
                    )
                    matched_detections.add(d)
                    matched_tracks.add(t)
                    
            # Create new tracks for unmatched detections
            for d, det in enumerate(detections):
                if d not in matched_detections:
                    track = TrackedObject(
                        self.next_id,
                        det['bbox'],
                        det['position_3d'],
                        det['class_id'],
                        det['confidence'],
                        self.process_noise,
                        self.measurement_noise
                    )
                    track.last_update_time = current_time
                    self.tracks.append(track)
                    self.next_id += 1
                    
        else:
            # No tracks or no detections
            for det in detections:
                track = TrackedObject(
                    self.next_id,
                    det['bbox'],
                    det['position_3d'],
                    det['class_id'],
                    det['confidence'],
                    self.process_noise,
                    self.measurement_noise
                )
                track.last_update_time = current_time
                self.tracks.append(track)
                self.next_id += 1
                
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Get confirmed tracks
        confirmed_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.age <= self.min_hits:
                confirmed_tracks.append(track.get_state())
                
        return confirmed_tracks
        
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
            
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Declare parameters for model and detection
        self.declare_parameter('model_path', '/home/john/ros2_ws/src/Triton/triton_bringup/weights/best.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('confidence_threshold_class_0', 0.65)  # Buoy
        self.declare_parameter('confidence_threshold_class_1', 0.4)  # Can
        self.declare_parameter('image_width', 480)
        self.declare_parameter('image_height', 480)
        
        # Declare parameters for topics
        self.declare_parameter(
            'image_topic',
            '/rgb',
            ParameterDescriptor(description='Input image topic')
        )
        self.declare_parameter(
            'output_topic',
            '/rgb/annotated',
            ParameterDescriptor(description='Output image topic with drawn bounding boxes')
        )
        self.declare_parameter(
            'detection_topic',
            '/yolo_detections',
            ParameterDescriptor(description='Detection2DArray topic with YOLO detections')
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
            'yaw_bias_deg',
            0.0,
            ParameterDescriptor(description='Fixed yaw bias in degrees to compensate camera misalignment')
        )
        self.declare_parameter(
            'marker_lifetime',
            1.0,
            ParameterDescriptor(description='Marker lifetime in seconds (0 for infinite)')
        )
        
        # Tracking parameters
        self.declare_parameter(
            'tracking_enabled',
            True,
            ParameterDescriptor(description='Enable Kalman filter tracking for marker stabilization')
        )
        self.declare_parameter(
            'max_tracking_age',
            5,
            ParameterDescriptor(description='Maximum frames to keep track without detection')
        )
        self.declare_parameter(
            'min_tracking_hits',
            3,
            ParameterDescriptor(description='Minimum hits before track is confirmed')
        )
        self.declare_parameter(
            'iou_threshold',
            0.3,
            ParameterDescriptor(description='Minimum IoU for track association')
        )
        self.declare_parameter(
            'process_noise',
            0.1,
            ParameterDescriptor(description='Kalman filter process noise')
        )
        self.declare_parameter(
            'measurement_noise',
            1.0,
            ParameterDescriptor(description='Kalman filter measurement noise')
        )
        self.declare_parameter(
            'track_id_offset',
            0,
            ParameterDescriptor(description='Offset added to track IDs to prevent conflicts between cameras')
        )
        
        # Get parameter values
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.confidence_threshold_class_0 = self.get_parameter('confidence_threshold_class_0').get_parameter_value().double_value
        self.confidence_threshold_class_1 = self.get_parameter('confidence_threshold_class_1').get_parameter_value().double_value
        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value
        
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
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
        self.yaw_bias_deg = self.get_parameter('yaw_bias_deg').get_parameter_value().double_value
        
        # Get tracking parameters
        self.tracking_enabled = self.get_parameter('tracking_enabled').get_parameter_value().bool_value
        self.max_tracking_age = self.get_parameter('max_tracking_age').get_parameter_value().integer_value
        self.min_tracking_hits = self.get_parameter('min_tracking_hits').get_parameter_value().integer_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.process_noise = self.get_parameter('process_noise').get_parameter_value().double_value
        self.measurement_noise = self.get_parameter('measurement_noise').get_parameter_value().double_value
        self.track_id_offset = self.get_parameter('track_id_offset').get_parameter_value().integer_value
        
        # Load YOLO model
        self.get_logger().info(f'Loading YOLOv11 model from {model_path}...')
        try:
            self.model = YOLO(model_path)
            self.get_logger().info('Model loaded successfully')
            # Get class names from model
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            raise
        
        # Setup CV Bridge
        self.cv_bridge = CvBridge()
        
        # Create publishers
        self.image_pub = self.create_publisher(Image, output_topic, 10)
        self.detection_pub = self.create_publisher(Detection2DArray, detection_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        
        # Create subscription
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        # Initialize tracker if enabled
        if self.tracking_enabled:
            self.tracker = ObjectTracker(
                max_age=self.max_tracking_age,
                min_hits=self.min_tracking_hits,
                iou_threshold=self.iou_threshold,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )
            self.previous_track_ids = set()
        else:
            self.tracker = None
        
        self.get_logger().info(f'YOLO Detector started')
        self.get_logger().info(f'Subscribing to image: {image_topic}')
        self.get_logger().info(f'Publishing annotated images to: {output_topic}')
        self.get_logger().info(f'Publishing detections to: {detection_topic}')
        self.get_logger().info(f'Publishing 3D markers to: {marker_topic}')
        self.get_logger().info(f'Camera parameters: cx={self.cx}, cy={self.cy}, fx={self.fx}, fy={self.fy}')
        self.get_logger().info(f'Projection parameters: fov={self.fov:.3f} rad, height={self.height}m, offset={self.offset}m')
        self.get_logger().info(f'Confidence thresholds: Class 0 (buoy)={self.confidence_threshold_class_0}, Class 1 (can)={self.confidence_threshold_class_1}')
        self.get_logger().info(f'Tracking: {"ENABLED" if self.tracking_enabled else "DISABLED"} (Kalman filter with process_noise={self.process_noise}, measurement_noise={self.measurement_noise})')
    
    def get_color_for_class(self, class_name):
        """Get a consistent color for each class"""
        # Handle both string and integer class identifiers
        if isinstance(class_name, str):
            if class_name.startswith("class_"):
                class_id = class_name.split("_")[1]
            else:
                # Try to extract numeric ID from the class name
                try:
                    class_id = str(int(class_name))
                except ValueError:
                    # If class name is like "buoy" or "can", map it
                    if "buoy" in class_name.lower():
                        class_id = "0"
                    elif "can" in class_name.lower():
                        class_id = "1"
                    else:
                        class_id = class_name
        else:
            class_id = str(class_name)
        
        # Assign colors based on class ID
        if class_id == "0":  # Buoy
            return (0, 128, 255)  # Orange in BGR format
        elif class_id == "1":  # Can
            return (0, 0, 255)    # Red in BGR format
        else:
            return (128, 128, 128)  # Gray for unknown classes
    
    def image_callback(self, msg):
        """Process image and generate detections with 3D markers"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Resize image if needed
            if cv_image.shape[1] != self.image_width or cv_image.shape[0] != self.image_height:
                cv_image_resized = cv2.resize(cv_image, (self.image_width, self.image_height))
            else:
                cv_image_resized = cv_image
            
            # Run YOLO detection with the minimum confidence threshold
            min_conf = min(self.confidence_threshold_class_0, self.confidence_threshold_class_1)
            results = self.model(cv_image_resized, conf=min_conf)
            
            # Create Detection2DArray message
            detection_array = Detection2DArray()
            detection_array.header = msg.header
            
            # Create marker array
            marker_array = MarkerArray()
            
            # Process detections if any
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Get scale factors for mapping back to original image size
                scale_x = cv_image.shape[1] / cv_image_resized.shape[1]
                scale_y = cv_image.shape[0] / cv_image_resized.shape[0]
                
                # Filter detections based on class-specific confidence thresholds
                valid_detections = []
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Apply class-specific confidence threshold
                    if class_id == 0 and confidence >= self.confidence_threshold_class_0:
                        valid_detections.append(i)
                    elif class_id == 1 and confidence >= self.confidence_threshold_class_1:
                        valid_detections.append(i)
                
                # Prepare arrays for vectorized 3D projection
                n_detections = len(valid_detections)
                if n_detections > 0:
                    self.get_logger().debug(f'Filtered {len(boxes)} detections to {n_detections} based on class-specific thresholds')
                center_x = np.zeros(n_detections)
                center_y = np.zeros(n_detections)
                size_y = np.zeros(n_detections)
                
                # Process each valid detection
                for idx, i in enumerate(valid_detections):
                    box = boxes[i]
                    # Get box coordinates (scaled back to original image size)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    # Calculate center and size
                    center_x[idx] = (x1 + x2) / 2
                    center_y[idx] = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    size_y[idx] = height
                    
                    # Create Detection2D message
                    detection = Detection2D()
                    detection.header = msg.header
                    
                    # Set bounding box
                    detection.bbox.center.position.x = center_x[idx]
                    detection.bbox.center.position.y = center_y[idx]
                    detection.bbox.size_x = width
                    detection.bbox.size_y = height
                    
                    # Get class information
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Create ObjectHypothesisWithPose
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(class_id)
                    hypothesis.hypothesis.score = confidence
                    detection.results.append(hypothesis)
                    
                    detection_array.detections.append(detection)
                    
                    # Get color for this class
                    color = self.get_color_for_class(class_name)
                    
                    # Draw bounding box on original image
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = f"{class_name}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw filled rectangle for label background
                    cv2.rectangle(cv_image, 
                                (int(x1), int(y1) - label_height - 4),
                                (int(x1) + label_width + 4, int(y1)),
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(cv_image, label,
                              (int(x1) + 2, int(y1) - 2),
                              font, font_scale,
                              (255, 255, 255),  # White text
                              thickness, cv2.LINE_AA)
                
                # Vectorized 3D projection
                bottom_y = center_y + size_y / 2
                pixels = np.column_stack([center_x, bottom_y])
                pixels_centered = pixels - np.array([self.cx, self.cy])
                angles = (self.fov/2) * pixels_centered / np.array([self.fx, self.fy])
                # Apply optional fixed yaw bias to correct small misalignment
                if self.yaw_bias_deg != 0.0:
                    angles[:, 0] = angles[:, 0] + (np.pi / 180.0) * self.yaw_bias_deg
                
                # Compute z and x for all points at once
                z_values = self.height / np.tan(angles[:, 1]) + self.offset
                x_values = z_values * np.tan(angles[:, 0])
                
                # Use tracker if enabled
                if self.tracking_enabled and self.tracker is not None:
                    # Prepare detections for tracker
                    tracker_detections = []
                    for idx, i in enumerate(valid_detections):
                        box = boxes[i]
                        tracker_detections.append({
                            'bbox': [box.xyxy[0][0].cpu().numpy() * scale_x,
                                    box.xyxy[0][1].cpu().numpy() * scale_y,
                                    box.xyxy[0][2].cpu().numpy() * scale_x,
                                    box.xyxy[0][3].cpu().numpy() * scale_y],
                            'position_3d': [x_values[idx], self.height, z_values[idx]],
                            'class_id': int(box.cls[0].cpu().numpy()),
                            'confidence': float(box.conf[0].cpu().numpy())
                        })
                    
                    # Update tracker and get confirmed tracks
                    current_time = self.get_clock().now()
                    confirmed_tracks = self.tracker.update(tracker_detections, current_time)
                    
                    # Get current track IDs
                    current_track_ids = set(track['id'] for track in confirmed_tracks)
                    
                    # Generate DELETE markers for lost tracks
                    lost_track_ids = self.previous_track_ids - current_track_ids
                    for track_id in lost_track_ids:
                        delete_marker = Marker()
                        delete_marker.header = msg.header
                        delete_marker.id = track_id + self.track_id_offset
                        delete_marker.ns = "tracked_detections"
                        delete_marker.action = Marker.DELETE
                        marker_array.markers.append(delete_marker)
                        self.get_logger().debug(f'Deleted track {track_id}')
                    
                    # Update previous track IDs
                    self.previous_track_ids = current_track_ids
                    
                    # Create markers from tracked objects
                    for i, track in enumerate(confirmed_tracks):
                        marker = Marker()
                        marker.header = msg.header
                        marker.id = track['id'] + self.track_id_offset  # Add offset for camera-specific IDs
                        marker.ns = "tracked_detections"
                        
                        # Use filtered position from Kalman filter
                        marker.pose.position.x = track['position'][0]
                        marker.pose.position.y = track['position'][1]
                        marker.pose.position.z = track['position'][2]
                        
                        # Get color based on class
                        class_name = self.class_names.get(track['class_id'], f"class_{track['class_id']}")
                        bbox_color = self.get_color_for_class(class_name)
                        marker.color.r = bbox_color[2] / 255.0  # BGR to RGB
                        marker.color.g = bbox_color[1] / 255.0
                        marker.color.b = bbox_color[0] / 255.0
                        marker.color.a = 1.0
                        
                        # Set marker type and scale based on class
                        if track['class_id'] == 0:  # Buoy
                            marker.type = Marker.CYLINDER
                            marker.scale.x = 0.4
                            marker.scale.y = 0.4
                            marker.scale.z = 0.8
                            
                            # Adjust position so cylinder extends upward from water surface
                            marker.pose.position.y = self.height / 2.0 + 0.7
                            
                            # No rotation needed - cylinder is already aligned with z-axis
                            marker.pose.orientation.x = 0.0
                            marker.pose.orientation.y = 0.0
                            marker.pose.orientation.z = 0.0
                            marker.pose.orientation.w = 1.0
                        else:  # Debris (can, cup, bottle, etc.)
                            marker.type = Marker.SPHERE
                            marker.scale.x = 0.3
                            marker.scale.y = 0.3
                            marker.scale.z = 0.3
                            
                            # No rotation needed for spheres
                            marker.pose.orientation.x = 0.0
                            marker.pose.orientation.y = 0.0
                            marker.pose.orientation.z = 0.0
                            marker.pose.orientation.w = 1.0
                        
                        # Set marker lifetime - ensure it's reasonable for non-tracking mode
                        if self.marker_lifetime > 0:
                            marker.lifetime = rclpy.duration.Duration(seconds=min(self.marker_lifetime, 0.5)).to_msg()
                        
                        marker.action = Marker.ADD
                        marker_array.markers.append(marker)
                        
                        self.get_logger().debug(f'Track {track["id"]}: {class_name} at ({track["position"][0]:.3f}, {track["position"][1]:.3f}, {track["position"][2]:.3f})')
                else:
                    # No tracking - use raw detections
                    for i, detection in enumerate(detection_array.detections):
                        # Get class information
                        class_id = detection.results[0].hypothesis.class_id
                        class_name = self.class_names.get(int(class_id), f"class_{class_id}")
                        
                        # Create 3D marker
                        marker = Marker()
                        marker.header = msg.header
                        marker.id = i
                        marker.ns = "detections"
                        
                        # Set position using pre-computed 3D coordinates
                        marker.pose.position.x = x_values[i]
                        marker.pose.position.y = self.height
                        marker.pose.position.z = z_values[i]
                        
                        # Get color from bounding box (BGR) and convert to RGB
                        bbox_color = self.get_color_for_class(class_name)
                        marker.color.r = bbox_color[2] / 255.0  # BGR to RGB
                        marker.color.g = bbox_color[1] / 255.0
                        marker.color.b = bbox_color[0] / 255.0
                        marker.color.a = 1.0
                        
                        # Set marker type and scale based on class
                        # Assuming class_id 3 is buoy in YOLO model (adjust as needed)
                        if class_id == '0':  # Buoy
                            marker.type = Marker.CYLINDER
                            marker.scale.x = 0.4
                            marker.scale.y = 0.4
                            marker.scale.z = 0.8
                            
                            # Adjust position so cylinder extends upward from water surface
                            marker.pose.position.y = self.height / 2.0 + 0.7
                            
                            # No rotation needed - cylinder is already aligned with z-axis
                            marker.pose.orientation.x = 0.0
                            marker.pose.orientation.y = 0.0
                            marker.pose.orientation.z = 0.0
                            marker.pose.orientation.w = 1.0
                        else:  # Debris (can, cup, bottle, etc.)
                            marker.type = Marker.SPHERE
                            marker.scale.x = 0.3
                            marker.scale.y = 0.3
                            marker.scale.z = 0.3
                            
                            # No rotation needed for spheres
                            marker.pose.orientation.x = 0.0
                            marker.pose.orientation.y = 0.0
                            marker.pose.orientation.z = 0.0
                            marker.pose.orientation.w = 1.0
                        
                        # Set marker lifetime - ensure it's reasonable for non-tracking mode
                        if self.marker_lifetime > 0:
                            marker.lifetime = rclpy.duration.Duration(seconds=min(self.marker_lifetime, 0.5)).to_msg()
                        
                        marker.action = Marker.ADD
                        marker_array.markers.append(marker)
                        
                        self.get_logger().debug(f'Created {class_name} marker at ({x_values[i]:.3f}, {self.height:.3f}, {z_values[i]:.3f})')
            
            # Publish detection array
            self.detection_pub.publish(detection_array)
            
            # Publish marker array
            if marker_array.markers:
                self.marker_pub.publish(marker_array)
                self.get_logger().debug(f'Published {len(marker_array.markers)} markers')
            
            # Convert back to ROS image and publish
            output_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
            output_msg.header = msg.header
            self.image_pub.publish(output_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
