#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from cv_bridge import CvBridge
import cv2


class KalmanFilter3D:
    """Kalman filter for 3D object tracking, specialized for stationary objects.

    State: [x, y, z, vx, vy, vz], but velocity is constrained toward 0.
    Allows per-update measurement covariance to reflect range-dependent uncertainty.
    """
    def __init__(self, process_noise=0.01, measurement_noise=1.0):
        self.state = np.zeros(6)
        self.F = np.eye(6)  # Constant-position model
        self.H = np.eye(3, 6)  # Observe position only

        # Process noise: small on position, near-zero on velocity (stationary prior)
        self.Q = np.eye(6) * process_noise
        self.Q[3:, 3:] = 1e-6

        # Default measurement noise (overridden per update if provided)
        self.R = np.eye(3) * measurement_noise

        # State covariance: start uncertain in position, low in velocity
        self.P = np.eye(6)
        self.P[:3, :3] *= 25.0
        self.P[3:, 3:] *= 1e-3

        self.initialized = False

    def initialize(self, x, y, z):
        self.state[:3] = [x, y, z]
        self.state[3:] = 0.0
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return
        # No motion model; keep state, grow uncertainty slightly
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Hard-constrain velocity toward zero
        self.state[3:] = 0.0

    def update(self, x, y, z, R_override=None):
        if not self.initialized:
            self.initialize(x, y, z)
            return
        z_meas = np.array([x, y, z])
        R = self.R if R_override is None else R_override
        # Innovation
        innov = z_meas - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innov
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self):
        return self.state[:3]

    def get_velocity(self):
        return self.state[3:]
        
    def get_position(self):
        """Get filtered position"""
        return self.state[:3]
        
    def get_velocity(self):
        """Get estimated velocity"""
        return self.state[3:]


class TrackedObject:
    """Represents a tracked object with Kalman filter"""
    def __init__(self, track_id, position, class_id, confidence, process_noise=0.1, measurement_noise=1.0):
        self.id = track_id
        self.class_id = class_id
        self.confidence = confidence
        self.kalman = KalmanFilter3D(process_noise, measurement_noise)
        self.kalman.initialize(*position)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.last_update_time = None
        self.source_cameras = set()  # Track which cameras have seen this object
        
    def predict(self, current_time):
        """Predict next state"""
        if self.last_update_time is not None:
            dt = (current_time - self.last_update_time).nanoseconds / 1e9
            self.kalman.predict(dt)
        self.age += 1
        self.time_since_update += 1
        
    def update(self, position, confidence, source_camera, current_time, R_override=None):
        """Update with new detection"""
        self.confidence = confidence
        if R_override is not None:
            self.kalman.update(*position, R_override=R_override)
        else:
            self.kalman.update(*position)
        self.hits += 1
        self.time_since_update = 0
        self.last_update_time = current_time
        # Handle comma-separated source cameras
        if ',' in source_camera:
            for cam in source_camera.split(','):
                self.source_cameras.add(cam)
        else:
            self.source_cameras.add(source_camera)
        
    def get_state(self):
        """Get current state for marker publishing"""
        pos = self.kalman.get_position()
        return {
            'id': self.id,
            'position': pos,
            'velocity': self.kalman.get_velocity(),
            'class_id': self.class_id,
            'confidence': self.confidence,
            'source_cameras': list(self.source_cameras)
        }


class GlobalTracker:
    """Global multi-object tracker that handles detections from multiple cameras"""
    def __init__(self, max_age=10, min_hits=3, distance_threshold=1.5,
                 process_noise=0.01, measurement_noise=1.0,
                 mahalanobis_threshold=3.5, range_noise_scale=0.02, depth_noise_factor=2.5,
                 class_distance_thresholds=None, class_mahalanobis_thresholds=None):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.mahalanobis_threshold = mahalanobis_threshold
        self.range_noise_scale = range_noise_scale
        self.depth_noise_factor = depth_noise_factor
        self.tracks = []
        self.next_id = 0
        # Position averaging window for stationary objects
        self.position_memory = {}  # track_id -> list of recent positions
        # Allow class-specific clustering distances (e.g. tighter for cans near each other)
        self.class_distance_thresholds = class_distance_thresholds or {}
        # Allow per-class gating to keep tightly packed debris from merging into one track
        self.class_mahalanobis_thresholds = class_mahalanobis_thresholds or {}

    def _measurement_covariance(self, position):
        """Build measurement covariance R based on range. Larger range => higher variance.
        Uses measurement_noise as base (in meters), and adds scale * r^2. Depth axis inflated.
        """
        pos = np.asarray(position)
        r = float(np.linalg.norm(pos))
        sigma_base = float(self.measurement_noise)
        sigma_lat = sigma_base + self.range_noise_scale * (r ** 2)
        sigma_y = max(0.2, 0.5 * sigma_lat)  # vertical usually better constrained
        sigma_depth = self.depth_noise_factor * sigma_lat
        R = np.diag([sigma_lat**2, sigma_y**2, sigma_depth**2])
        return R
        
    def update(self, detections, current_time):
        """
        Update tracker with new detections from all cameras
        detections: list of dicts with keys: position, class_id, confidence, source_camera
        """
        # Predict all tracks
        for track in self.tracks:
            track.predict(current_time)
            
        # Group detections by approximate location to handle multi-camera overlap.
        # Thresholds are class-aware so debris can effectively skip merging when set near zero.
        detection_groups = []
        used_detections = set()
        
        for i, det1 in enumerate(detections):
            if i in used_detections:
                continue
                
            group = [i]
            used_detections.add(i)
            
            for j, det2 in enumerate(detections):
                if j <= i or j in used_detections:
                    continue
                    
                # Check if detections are close and same class
                if det1['class_id'] != det2['class_id']:
                    continue

                dist = np.linalg.norm(np.array(det1['position']) - np.array(det2['position']))
                threshold = self.class_distance_thresholds.get(
                    det1['class_id'],
                    self.distance_threshold
                )

                # Threshold <= 0 disables grouping for this class (treat every detection as unique)
                if threshold <= 0.0:
                    continue

                # Use configured threshold to decide whether detections belong together
                if dist < threshold:
                    group.append(j)
                    used_detections.add(j)
                    
            detection_groups.append(group)
        
        # Merge grouped detections by (variance-)weighted averaging positions
        merged_detections = []
        for group in detection_groups:
            if len(group) == 1:
                merged_detections.append(detections[group[0]])
            else:
                # Weighted average using inverse variance from measurement covariance
                positions = [np.array(detections[idx]['position']) for idx in group]
                variances = []
                for idx in group:
                    R = self._measurement_covariance(detections[idx]['position'])
                    variances.append(np.diag(R))
                variances = np.array(variances)  # shape (N,3)
                # Avoid division by zero
                variances = np.clip(variances, 1e-6, None)
                weights = 1.0 / variances  # shape (N,3)
                weights_sum = np.sum(weights, axis=0)
                weighted_pos_sum = np.sum(np.array(positions) * weights, axis=0)
                avg_position = weighted_pos_sum / weights_sum
                
                # Take highest confidence
                confidences = [detections[idx]['confidence'] for idx in group]
                best_idx = group[np.argmax(confidences)]
                
                # Combine source cameras
                source_cameras = []
                for idx in group:
                    cam = detections[idx]['source_camera']
                    if cam not in source_cameras:
                        source_cameras.append(cam)
                
                merged_det = {
                    'position': avg_position.tolist(),
                    'class_id': detections[best_idx]['class_id'],
                    'confidence': max(confidences),
                    'source_camera': ','.join(source_cameras)
                }
                merged_detections.append(merged_det)
        
        # Now associate merged detections to tracks
        if len(self.tracks) > 0 and len(merged_detections) > 0:
            # Calculate distance matrix
            distance_matrix = np.zeros((len(merged_detections), len(self.tracks)))
            for d, det in enumerate(merged_detections):
                for t, track in enumerate(self.tracks):
                    track_pos = track.kalman.get_position()
                    det_pos = np.array(det['position'])
                    # Mahalanobis distance using track covariance + measurement covariance
                    R = self._measurement_covariance(det_pos)
                    P_pos = track.kalman.P[:3, :3]
                    S = P_pos + R
                    try:
                        innov = det_pos - track_pos
                        d2 = float(innov.T @ np.linalg.inv(S) @ innov)
                        distance = np.sqrt(max(d2, 0.0))
                    except np.linalg.LinAlgError:
                        # Fallback to Euclidean if S not invertible
                        distance = float(np.linalg.norm(det_pos - track_pos))
                    # Penalize class mismatch
                    if det['class_id'] != track.class_id:
                        distance += 10.0
                    distance_matrix[d, t] = distance
                    
            # Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(distance_matrix)
            
            # Update matched tracks
            matched_detections = set()
            matched_tracks = set()
            
            for d, t in zip(row_ind, col_ind):
                det = merged_detections[d]
                # Accept match only if inside Mahalanobis gate
                gate = self.class_mahalanobis_thresholds.get(
                    self.tracks[t].class_id,
                    self.mahalanobis_threshold
                )

                if gate <= 0.0:
                    continue  # Explicitly disable matching for this class if requested

                if distance_matrix[d, t] < gate:
                    R = self._measurement_covariance(det['position'])
                    self.tracks[t].update(
                        det['position'],
                        det['confidence'],
                        det['source_camera'],
                        current_time,
                        R_override=R
                    )
                    matched_detections.add(d)
                    matched_tracks.add(t)
                    
            # Create new tracks for unmatched detections
            for d, det in enumerate(merged_detections):
                if d not in matched_detections:
                    track = TrackedObject(
                        self.next_id,
                        det['position'],
                        det['class_id'],
                        det['confidence'],
                        self.process_noise,
                        self.measurement_noise
                    )
                    track.last_update_time = current_time
                    # Handle comma-separated source cameras
                    for cam in det['source_camera'].split(','):
                        track.source_cameras.add(cam)
                    self.tracks.append(track)
                    # Initialize position memory for stationary object tracking
                    self.position_memory[self.next_id] = [det['position']]
                    self.next_id += 1
                    
        else:
            # No tracks or no detections - create new tracks for all detections
            for det in merged_detections:
                track = TrackedObject(
                    self.next_id,
                    det['position'],
                    det['class_id'],
                    det['confidence'],
                    self.process_noise,
                    self.measurement_noise
                )
                track.last_update_time = current_time
                # Handle comma-separated source cameras
                for cam in det['source_camera'].split(','):
                    track.source_cameras.add(cam)
                self.tracks.append(track)
                # Initialize position memory for stationary object tracking
                self.position_memory[self.next_id] = [det['position']]
                self.next_id += 1
                
        # Remove dead tracks
        dead_track_ids = []
        new_tracks = []
        for track in self.tracks:
            if track.time_since_update < self.max_age:
                new_tracks.append(track)
            else:
                dead_track_ids.append(track.id)
        self.tracks = new_tracks
        
        # Clean up position memory for dead tracks
        for track_id in dead_track_ids:
            if track_id in self.position_memory:
                del self.position_memory[track_id]
        
        # Get confirmed tracks
        confirmed_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.age <= self.min_hits:
                confirmed_tracks.append(track.get_state())
                
        return confirmed_tracks


class OrienteeringNode(Node):
    def __init__(self):
        super().__init__('orienteering_node')
        
        # Declare and get the marker topics parameter
        self.declare_parameter('marker_topics', [
            '/cubemap_front/detection/markers',
            '/cubemap_back/detection/markers',
            '/cubemap_left/detection/markers',
            '/cubemap_right/detection/markers'
        ])
        self.marker_topics = self.get_parameter('marker_topics').get_parameter_value().string_array_value
        
        # Declare output topic parameter
        self.declare_parameter('output_topic', '/combined_markers')
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # Declare image topics parameter for annotated images
        self.declare_parameter('image_topics', [
            '/cubemap_left/detection/image',
            '/cubemap_front/detection/image',
            '/cubemap_right/detection/image',
            '/cubemap_back/detection/image'
        ])
        self.image_topics = self.get_parameter('image_topics').get_parameter_value().string_array_value
        
        # Declare stitched image output topic
        self.declare_parameter('stitched_image_topic', '/stitched_detections')
        self.stitched_image_topic = self.get_parameter('stitched_image_topic').get_parameter_value().string_value
        
        # Declare output frame parameter
        self.declare_parameter('output_frame', 'map')
        self.output_frame = self.get_parameter('output_frame').get_parameter_value().string_value
        
        # Declare sync time tolerance parameter (in seconds)
        self.declare_parameter('sync_tolerance', 0.1)
        self.sync_tolerance = self.get_parameter('sync_tolerance').get_parameter_value().double_value
        
        # Declare transform timeout parameter
        self.declare_parameter('transform_timeout', 0.1)
        self.transform_timeout = self.get_parameter('transform_timeout').get_parameter_value().double_value
        
        # Declare startup wait time parameter
        self.declare_parameter('startup_wait_time', 5.0)
        self.startup_wait_time = self.get_parameter('startup_wait_time').get_parameter_value().double_value
        
        # Declare tracking parameters
        self.declare_parameter('tracking_enabled', True)
        self.tracking_enabled = self.get_parameter('tracking_enabled').get_parameter_value().bool_value
        
        self.declare_parameter('max_tracking_age', 10)
        self.max_tracking_age = self.get_parameter('max_tracking_age').get_parameter_value().integer_value
        
        self.declare_parameter('min_tracking_hits', 3)
        self.min_tracking_hits = self.get_parameter('min_tracking_hits').get_parameter_value().integer_value
        
        self.declare_parameter('distance_threshold', 1.5)
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value

        self.declare_parameter('process_noise', 0.1)
        self.process_noise = self.get_parameter('process_noise').get_parameter_value().double_value

        self.declare_parameter('measurement_noise', 1.0)
        self.measurement_noise = self.get_parameter('measurement_noise').get_parameter_value().double_value

        # Allow class-specific clustering thresholds (useful when objects are tightly grouped)
        self.declare_parameter('debris_distance_threshold', 0.0)
        self.debris_distance_threshold = self.get_parameter('debris_distance_threshold').get_parameter_value().double_value
        self.class_distance_thresholds = {}
        self.class_distance_thresholds[1] = self.debris_distance_threshold

        # Allow class-specific gating for tracker association to avoid merging neighbouring debris
        self.declare_parameter('debris_mahalanobis_threshold', 0.6)
        self.debris_mahalanobis_threshold = self.get_parameter('debris_mahalanobis_threshold').get_parameter_value().double_value
        self.class_mahalanobis_thresholds = {}
        self.class_mahalanobis_thresholds[1] = self.debris_mahalanobis_threshold

        # Toggle optional visualization of tracking labels above markers
        self.declare_parameter('show_track_labels', False)
        self.show_track_labels = self.get_parameter('show_track_labels').get_parameter_value().bool_value
        
        # Declare color coding parameter
        self.declare_parameter('color_code', False)
        self.color_code = self.get_parameter('color_code').get_parameter_value().bool_value
        
        self.get_logger().info(f'Subscribing to marker topics: {self.marker_topics}')
        self.get_logger().info(f'Publishing combined markers to: {self.output_topic}')
        self.get_logger().info(f'Subscribing to image topics: {self.image_topics}')
        self.get_logger().info(f'Publishing stitched image to: {self.stitched_image_topic}')
        self.get_logger().info(f'Transforming markers to frame: {self.output_frame}')
        self.get_logger().info(f'Global tracking: {"ENABLED" if self.tracking_enabled else "DISABLED"}')
        
        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize CV Bridge
        self.cv_bridge = CvBridge()
        
        # Create publisher for combined markers
        self.combined_publisher = self.create_publisher(MarkerArray, self.output_topic, 10)
        
        # Create publisher for stitched image
        self.stitched_image_pub = self.create_publisher(Image, self.stitched_image_topic, 10)
        
        # Store latest messages from each topic
        self.latest_markers = {}
        self.latest_images = {}
        
        # Store cached transforms for each frame
        self.cached_transforms = {}
        
        # Track previous marker counts for each topic to detect when markers disappear
        self.previous_marker_counts = {}
        
        # Track startup time
        self.startup_time = self.get_clock().now()
        self.startup_complete = False
        
        # Initialize global tracker if enabled
        if self.tracking_enabled:
            # Optional advanced tracking parameters
            self.declare_parameter('mahalanobis_threshold', 3.5)
            self.declare_parameter('range_noise_scale', 0.02)
            self.declare_parameter('depth_noise_factor', 2.5)

            mahal = self.get_parameter('mahalanobis_threshold').get_parameter_value().double_value
            rn_scale = self.get_parameter('range_noise_scale').get_parameter_value().double_value
            depth_factor = self.get_parameter('depth_noise_factor').get_parameter_value().double_value

            self.tracker = GlobalTracker(
                max_age=self.max_tracking_age,
                min_hits=self.min_tracking_hits,
                distance_threshold=self.distance_threshold,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                mahalanobis_threshold=mahal,
                range_noise_scale=rn_scale,
                depth_noise_factor=depth_factor,
                class_distance_thresholds=self.class_distance_thresholds,
                class_mahalanobis_thresholds=self.class_mahalanobis_thresholds
            )
            self.previous_track_ids = set()
        else:
            self.tracker = None
        
        # Create subscribers for each marker topic
        self.subscribers = []
        for topic in self.marker_topics:
            sub = self.create_subscription(
                MarkerArray,
                topic,
                lambda msg, t=topic: self.marker_callback(msg, t),
                10
            )
            self.subscribers.append(sub)
            self.latest_markers[topic] = None
            self.previous_marker_counts[topic] = 0
        
        # Create subscribers for each image topic
        self.image_subscribers = []
        for topic in self.image_topics:
            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, t=topic: self.image_callback(msg, t),
                10
            )
            self.image_subscribers.append(sub)
            self.latest_images[topic] = None
        
        # Create timer for periodic publishing of combined markers and stitched image
        self.timer = self.create_timer(0.1, self.publish_combined_data)  # 10Hz
        
        self.get_logger().info(f'Orienteering node initialized. Waiting {self.startup_wait_time}s for transforms to stabilize...')

    def get_direction_colors(self, source_camera, class_id):
        """Get direction-specific colors for markers when color_code is enabled"""
        if not self.color_code:
            # Use default colors when color coding is disabled
            if class_id == 0:  # Buoy
                return (1.0, 0.65, 0.0)  # Orange
            else:  # Debris
                return (1.0, 0.0, 0.0)  # Red
        
        # Direction-specific colors when color_code is enabled
        direction_colors = {
            'cubemap_front': {
                0: (0.0, 0.8, 0.0),    # Green for buoys
                1: (0.4, 1.0, 0.4)     # Light green for debris
            },
            'cubemap_back': {
                0: (0.0, 0.4, 1.0),    # Blue for buoys  
                1: (0.4, 0.7, 1.0)     # Light blue for debris
            },
            'cubemap_left': {
                0: (1.0, 0.0, 0.0),    # Red for buoys
                1: (1.0, 0.4, 0.6)     # Light red/pink for debris
            },
            'cubemap_right': {
                0: (1.0, 0.8, 0.0),    # Yellow/orange for buoys
                1: (1.0, 1.0, 0.4)     # Light yellow for debris
            }
        }
        
        # Default to original colors if source not found
        colors = direction_colors.get(source_camera, {
            0: (1.0, 0.5, 0.0),  # Orange
            1: (1.0, 0.0, 0.0)   # Red
        })
        
        return colors.get(class_id, (1.0, 0.5, 0.0))

    def marker_callback(self, msg, topic):
        """Store the latest marker array from each topic"""
        self.latest_markers[topic] = msg
        
    def image_callback(self, msg, topic):
        """Store the latest image from each topic"""
        self.latest_images[topic] = msg
        
    def transform_marker(self, marker, target_frame):
        """Transform a marker to the target frame using cached TF2 transforms"""
        if marker.header.frame_id == target_frame:
            # Already in target frame
            return marker
        
        source_frame = marker.header.frame_id
        cache_key = f"{source_frame}->{target_frame}"
        
        # First, try to get a transform at the current time (latest available)
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),  # Latest available transform
                timeout=rclpy.duration.Duration(seconds=self.transform_timeout)
            )
            self.cached_transforms[cache_key] = transform
            
        except TransformException as e:
            # If that fails, check if we have a cached transform
            if cache_key in self.cached_transforms:
                self.get_logger().debug(f'Using cached transform for {cache_key} (latest lookup failed: {e})')
                transform = self.cached_transforms[cache_key]
            else:
                # No transform available at all
                self.get_logger().warning(f'No transform available for {cache_key}: {e}')
                return None
        
        # Apply the transform
        try:
            # Create a PoseStamped from the marker pose
            pose_stamped = PoseStamped()
            pose_stamped.header = marker.header
            # Update timestamp to match the transform time to avoid extrapolation
            pose_stamped.header.stamp = transform.header.stamp
            pose_stamped.pose = marker.pose
            
            # Transform the pose
            transformed_pose_stamped = tf2_geometry_msgs.do_transform_pose_stamped(
                pose_stamped, 
                transform
            )
            
            # Create a new marker with the transformed pose
            transformed_marker = copy.deepcopy(marker)
            transformed_marker.header.frame_id = target_frame
            transformed_marker.header.stamp = self.get_clock().now().to_msg()
            transformed_marker.pose = transformed_pose_stamped.pose
            
            return transformed_marker
            
        except Exception as e:
            self.get_logger().error(f'Failed to apply transform: {e}')
            return None
        
    def publish_combined_data(self):
        """Combine and publish both markers and stitched images"""
        self.publish_combined_markers()
        self.publish_stitched_image()
        
    def publish_stitched_image(self):
        """Stitch all camera images into a single panoramic view"""
        # Check if we have all images
        images = []
        for topic in self.image_topics:
            if topic in self.latest_images and self.latest_images[topic] is not None:
                try:
                    cv_image = self.cv_bridge.imgmsg_to_cv2(self.latest_images[topic], "bgr8")
                    images.append(cv_image)
                except Exception as e:
                    self.get_logger().error(f'Failed to convert image from {topic}: {e}')
                    return
            else:
                # Missing image, skip stitching this frame
                return
        
        if len(images) == 4:  # We have all 4 images
            # All images should be the same size (960x960)
            # Arrange as [left, front, right, back]
            stitched = np.hstack(images)
            
            # Convert back to ROS message
            try:
                stitched_msg = self.cv_bridge.cv2_to_imgmsg(stitched, "bgr8")
                stitched_msg.header.stamp = self.get_clock().now().to_msg()
                stitched_msg.header.frame_id = "camera_frame"
                self.stitched_image_pub.publish(stitched_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish stitched image: {e}')
        
    def publish_combined_markers(self):
        """Combine all latest markers and publish"""
        # Check if startup wait time has passed
        if not self.startup_complete:
            elapsed = (self.get_clock().now() - self.startup_time).nanoseconds / 1e9
            if elapsed < self.startup_wait_time:
                return  # Skip publishing during startup wait
            else:
                self.startup_complete = True
                self.get_logger().info('Startup wait complete. Beginning marker transformation.')
        
        combined_array = MarkerArray()
        
        if self.tracking_enabled and self.tracker is not None:
            # Collect all detections from all cameras
            all_detections = []
            
            for topic, marker_array in self.latest_markers.items():
                if marker_array is None or not marker_array.markers:
                    continue
                    
                source_camera = topic.split('/')[-2]  # e.g., 'cubemap_front'
                
                for marker in marker_array.markers:
                    # Skip DELETE markers - we handle deletion through tracking
                    if marker.action == Marker.DELETE:
                        continue
                        
                    # Transform marker to output frame
                    transformed_marker = self.transform_marker(marker, self.output_frame)
                    
                    if transformed_marker is None:
                        continue
                    
                    # Extract position and class information
                    position = [
                        transformed_marker.pose.position.x,
                        transformed_marker.pose.position.y,
                        transformed_marker.pose.position.z
                    ]
                    
                    # Determine class ID from marker properties
                    class_id = 0  # Default to buoy
                    if transformed_marker.type == Marker.SPHERE:
                        class_id = 1  # Can/debris
                    
                    # Extract confidence from color alpha (if available)
                    confidence = transformed_marker.color.a
                    
                    all_detections.append({
                        'position': position,
                        'class_id': class_id,
                        'confidence': confidence,
                        'source_camera': source_camera
                    })
            
            # Update global tracker
            current_time = self.get_clock().now()
            tracked_objects = self.tracker.update(all_detections, current_time)
            
            # Get current track IDs
            current_track_ids = set(track['id'] for track in tracked_objects)
            
            # Log detection and tracking info
            if len(all_detections) > 0 or len(tracked_objects) > 0:
                self.get_logger().debug(f'Raw detections: {len(all_detections)}, Tracked objects: {len(tracked_objects)}')
                # Log source cameras for each detection
                camera_counts = {}
                for det in all_detections:
                    cam = det['source_camera']
                    camera_counts[cam] = camera_counts.get(cam, 0) + 1
                if camera_counts:
                    self.get_logger().debug(f'Detections per camera: {camera_counts}')
            
            # Generate DELETE markers for lost tracks
            lost_track_ids = self.previous_track_ids - current_track_ids
            for track_id in lost_track_ids:
                # Delete main marker
                delete_marker = Marker()
                delete_marker.header.frame_id = self.output_frame
                delete_marker.header.stamp = self.get_clock().now().to_msg()
                delete_marker.id = track_id
                delete_marker.ns = "global_tracked_objects"
                delete_marker.action = Marker.DELETE
                combined_array.markers.append(delete_marker)

                if self.show_track_labels:
                    # Delete text label marker
                    delete_text_marker = Marker()
                    delete_text_marker.header.frame_id = self.output_frame
                    delete_text_marker.header.stamp = self.get_clock().now().to_msg()
                    delete_text_marker.id = track_id + 10000
                    delete_text_marker.ns = "global_tracked_labels"
                    delete_text_marker.action = Marker.DELETE
                    combined_array.markers.append(delete_text_marker)
                
                self.get_logger().debug(f'Deleted global track {track_id}')
            
            # Update previous track IDs
            self.previous_track_ids = current_track_ids
            
            # Create markers from tracked objects
            for track in tracked_objects:
                marker = Marker()
                marker.header.frame_id = self.output_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = track['id']
                marker.ns = "global_tracked_objects"
                
                # Set position from Kalman filter
                marker.pose.position.x = track['position'][0]
                marker.pose.position.y = track['position'][1]
                marker.pose.position.z = track['position'][2]
                
                # Determine primary source camera for color coding
                primary_camera = track['source_cameras'][0] if track['source_cameras'] else 'cubemap_front'
                
                # Get direction-specific colors
                color = self.get_direction_colors(primary_camera, track['class_id'])
                
                # Set marker type and color based on class
                if track['class_id'] == 0:  # Buoy
                    marker.type = Marker.CYLINDER
                    marker.scale.x = 0.7
                    marker.scale.y = 0.7
                    marker.scale.z = 0.9
                    marker.color.r = color[0]
                    marker.color.g = color[1]
                    marker.color.b = color[2]
                    marker.color.a = 1.0
                    
                    # No rotation needed - cylinder is already aligned with z-axis
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.pose.position.z -= marker.scale.z/2  # Move down by full height
                else:  # Can/debris
                    marker.type = Marker.SPHERE
                    marker.scale.x = 0.3
                    marker.scale.y = 0.3
                    marker.scale.z = 0.3
                    marker.color.r = color[0]
                    marker.color.g = color[1]
                    marker.color.b = color[2]
                    marker.color.a = 1.0
                    
                    # No rotation needed for spheres
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                
                marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                marker.action = Marker.ADD

                combined_array.markers.append(marker)

                if self.show_track_labels:
                    # Add text marker with track info
                    text_marker = Marker()
                    text_marker.header = marker.header
                    text_marker.id = track['id'] + 10000  # Offset for text markers
                    text_marker.ns = "global_tracked_labels"
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.text = f"ID: {track['id']}\nCams: {', '.join(track['source_cameras'])}"
                    text_marker.pose = marker.pose
                    # text_marker.pose.position.z += 0.5  # Offset above marker
                    text_marker.scale.z = 0.2
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    text_marker.color.a = 1.0
                    text_marker.lifetime = marker.lifetime
                    text_marker.action = Marker.ADD
                    combined_array.markers.append(text_marker)
                
                self.get_logger().debug(f'Global track {track["id"]}: class={track["class_id"]}, pos=({track["position"][0]:.2f}, {track["position"][1]:.2f}, {track["position"][2]:.2f}), cameras={track["source_cameras"]}')
        
        else:
            # Non-tracking mode - just combine and transform markers
            marker_id_offset = 0
            current_marker_counts = {}
            
            # Combine markers from all topics
            for topic, marker_array in self.latest_markers.items():
                # Track current marker count
                current_count = len(marker_array.markers) if marker_array is not None else 0
                current_marker_counts[topic] = current_count
                
                # Check if markers disappeared (had markers before but not now)
                if current_count == 0 and self.previous_marker_counts.get(topic, 0) > 0:
                    # Create DELETE markers for this topic's namespace
                    source_prefix = topic.split('/')[-2]  # e.g., 'cubemap_front' or 'cubemap_back'
                    for i in range(self.previous_marker_counts[topic]):
                        delete_marker = Marker()
                        delete_marker.header.frame_id = self.output_frame
                        delete_marker.header.stamp = self.get_clock().now().to_msg()
                        delete_marker.ns = f"{source_prefix}/detections"
                        delete_marker.id = marker_id_offset + i
                        delete_marker.action = Marker.DELETE
                        combined_array.markers.append(delete_marker)
                    self.get_logger().debug(f'Clearing {self.previous_marker_counts[topic]} markers from {topic}')
                
                if marker_array is None or not marker_array.markers:
                    # Update offset even for empty topics to maintain consistent IDs
                    if topic in self.previous_marker_counts:
                        marker_id_offset += self.previous_marker_counts[topic]
                    continue
                    
                for marker in marker_array.markers:
                    # Transform marker to output frame
                    transformed_marker = self.transform_marker(marker, self.output_frame)
                    
                    if transformed_marker is None:
                        # Skip markers that couldn't be transformed
                        continue
                    
                    # Apply colors in non-tracking mode
                    source_camera = topic.split('/')[1]  # Extract 'cubemap_front' from '/cubemap_front/detection/markers'
                    
                    # Determine class from marker type
                    class_id = 0 if transformed_marker.type == Marker.CYLINDER else 1
                    
                    # Get colors (direction-specific if color_code=True, default if False)
                    color = self.get_direction_colors(source_camera, class_id)
                    
                    # Apply the colors
                    transformed_marker.color.r = color[0]
                    transformed_marker.color.g = color[1] 
                    transformed_marker.color.b = color[2]
                    
                    # Scale cylinder obstacles and move down z-axis
                    if transformed_marker.type == Marker.CYLINDER:  # Buoy
                        transformed_marker.scale.x = 0.7
                        transformed_marker.scale.y = 0.7
                        transformed_marker.scale.z = 0.9
                        transformed_marker.pose.position.z -= transformed_marker.scale.z/2  # Move down by full height
                    
                    # Assign unique ID to avoid conflicts
                    transformed_marker.id = marker_id_offset + marker.id
                    
                    # Add namespace to distinguish source
                    if not transformed_marker.ns:
                        transformed_marker.ns = topic.replace('/', '_').strip('_')
                    else:
                        # Prepend topic info to existing namespace
                        source_prefix = topic.split('/')[-2]  # e.g., 'cubemap_front' or 'cubemap_back'
                        transformed_marker.ns = f"{source_prefix}/{transformed_marker.ns}"
                    
                    combined_array.markers.append(transformed_marker)
                
                # Update offset for next topic's markers
                if marker_array.markers:
                    max_id = max(m.id for m in marker_array.markers)
                    marker_id_offset += max_id + 1
            
            # Update previous marker counts
            self.previous_marker_counts = current_marker_counts
        
        # Publish combined markers if we have any (including DELETE markers)
        if combined_array.markers:
            self.combined_publisher.publish(combined_array)
            

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = OrienteeringNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
