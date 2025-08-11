#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math
from rcl_interfaces.msg import SetParametersResult


class EquirectangularNode(Node):
    def __init__(self, enable_calibration=False):
        super().__init__('equirectangular_node')

        self.params_changed = True
        
        # Declare parameters with default values from YAML
        self.declare_parameters(
            namespace='',
            parameters=[
                ('cx_offset', 0.0),
                ('cy_offset', 0.0),
                ('crop_size', 960),
                ('translation', [0.0, 0.0, -0.105]),
                ('rotation_deg', [-0.5, 0.0, 1.1]),
                ('gpu', True),
                ('out_width', 1920),
                ('out_height', 960)
            ]
        )
        
        # Load parameters from ROS parameter server
        self.load_parameters()
        
        # Initialize GPU settings
        self.use_cuda = torch.cuda.is_available() and getattr(self, 'gpu_enabled', True)
        self.get_logger().info(f"GPU acceleration: requested={getattr(self, 'gpu_enabled', True)}, available={torch.cuda.is_available()}, using={self.use_cuda}")
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.calibration_mode = enable_calibration
        self.maps_initialized = False
        
        self.img_height: int | None = None
        self.img_width: int | None = None
        
        # Force initial update in calibration mode
        if self.calibration_mode:
            self.params_changed = True
        
        self.last_front_img: np.ndarray | None = None
        self.last_back_img: np.ndarray | None = None
        self.original_front_img: np.ndarray | None = None
        self.original_back_img: np.ndarray | None = None

        # Precomputed masks and kernels
        self.front_mask: torch.Tensor | None = None
        self.back_mask: torch.Tensor | None = None
        self.front_mask_np: np.ndarray | None = None
        self.back_mask_np: np.ndarray | None = None
        self.blend_kernel: np.ndarray = np.ones((5, 5), np.uint8)
        self.front_edge: np.ndarray | None = None
        self.front_distance: np.ndarray | None = None

        # GPU specific masks and grids
        self.front_grid: torch.Tensor | None = None
        self.back_grid: torch.Tensor | None = None
        self.front_mask_gpu: torch.Tensor | None = None
        self.back_mask_gpu: torch.Tensor | None = None
        self.edge_mask_gpu: torch.Tensor | None = None
        self.blend_weight_gpu: torch.Tensor | None = None
        
        self.add_on_set_parameters_callback(self.parameters_callback)
        self.update_camera_parameters() # Initialize camera parameters
        
        self.bridge = CvBridge()
        
        # Configure QoS for reliable communication with buffer size 1
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE
        )
        
        self.dual_fisheye_sub = self.create_subscription(
            Image, '/dual_fisheye/image', self.image_callback, qos)
        self.equirect_pub = self.create_publisher(
            Image, '/equirectangular/image', qos)
        
        if self.calibration_mode:
            self.get_logger().info("Calibration mode enabled")
            self.setup_calibration_ui()
    
    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        try:
            self.cx_offset = self.get_parameter('cx_offset').get_parameter_value().double_value
            self.cy_offset = self.get_parameter('cy_offset').get_parameter_value().double_value
            self.crop_size = self.get_parameter('crop_size').get_parameter_value().integer_value
            self.out_width = self.get_parameter('out_width').get_parameter_value().integer_value
            self.out_height = self.get_parameter('out_height').get_parameter_value().integer_value
            self.gpu_enabled = self.get_parameter('gpu').get_parameter_value().bool_value
            
            translation = self.get_parameter('translation').get_parameter_value().double_array_value
            self.tx, self.ty, self.tz = translation
            
            rotation_deg = self.get_parameter('rotation_deg').get_parameter_value().double_array_value
            self.roll = math.radians(rotation_deg[0])
            self.pitch = math.radians(rotation_deg[1])
            self.yaw = math.radians(rotation_deg[2])
            
            self.get_logger().info(f"Loaded parameters from ROS parameter server")
            self.get_logger().info(f"  Crop size: {self.crop_size}")
            self.get_logger().info(f"  Center offset: ({self.cx_offset}, {self.cy_offset})")
            self.get_logger().info(f"  Translation: [{self.tx}, {self.ty}, {self.tz}]")
            self.get_logger().info(f"  Rotation (deg): {rotation_deg}")
            self.get_logger().info(f"  Output size: {self.out_width}x{self.out_height}")
            self.get_logger().info(f"  GPU enabled: {self.gpu_enabled}")
        except Exception as e:
            self.get_logger().error(f"Error loading parameters: {e}")
            # Set defaults if parameter loading fails
            self.gpu_enabled = True
            raise
    
    def save_calibration(self):
        """Save current calibration parameters to ROS parameter server"""
        try:
            self.set_parameters([
                Parameter('cx_offset', Parameter.Type.DOUBLE, self.cx_offset),
                Parameter('cy_offset', Parameter.Type.DOUBLE, self.cy_offset),
                Parameter('crop_size', Parameter.Type.INTEGER, self.crop_size),
                Parameter('translation', Parameter.Type.DOUBLE_ARRAY, [self.tx, self.ty, self.tz]),
                Parameter('rotation_deg', Parameter.Type.DOUBLE_ARRAY, [
                    math.degrees(self.roll),
                    math.degrees(self.pitch),
                    math.degrees(self.yaw)
                ])
            ])
            
            # Print parameters in YAML format for copy-pasting
            print("\n" + "="*50)
            print("CALIBRATION PARAMETERS (YAML FORMAT)")
            print("="*50)
            print("equirectangular_node:")
            print("  ros__parameters:")
            print(f"    cx_offset: {self.cx_offset}")
            print(f"    cy_offset: {self.cy_offset}")
            print(f"    crop_size: {self.crop_size}")
            print(f"    translation: [{self.tx}, {self.ty}, {self.tz}]")
            print(f"    rotation_deg: [{math.degrees(self.roll)}, {math.degrees(self.pitch)}, {math.degrees(self.yaw)}]")
            print(f"    gpu: {self.gpu_enabled}")
            print(f"    out_width: {self.out_width}")
            print(f"    out_height: {self.out_height}")
            print("="*50 + "\n")
            
            self.get_logger().info("Parameters saved to ROS parameter server and printed above")
            return True
        except Exception as e:
            self.get_logger().error(f"Error saving parameters: {e}")
            return False

    def parameters_callback(self, params):
        """Parameter update callback for dynamic reconfiguration"""
        update_needed = False
        
        for param in params:
            # Check if a camera parameter was changed
            if param.name in ['cx_offset', 'cy_offset', 'crop_size', 'translation', 'rotation_deg',
                             'out_width', 'out_height', 'gpu']:
                update_needed = True
                
        if update_needed:
            self.load_parameters()
            self.update_camera_parameters()
            
        return SetParametersResult(successful=True)
    
    def update_trackbar_positions(self):
        """Update trackbar positions to match current parameter values"""
        if not hasattr(self, 'control_window'):
            return
            
        # Update trackbar positions without triggering callbacks
        cv2.setTrackbarPos("CX Offset [-100,100]", self.control_window, int(self.cx_offset) + 100)
        cv2.setTrackbarPos("CY Offset [-100,100]", self.control_window, int(self.cy_offset) + 100)
        cv2.setTrackbarPos("Crop Size", self.control_window, self.crop_size)
        cv2.setTrackbarPos("TX [-0.5,0.5]", self.control_window, int(self.tx * 1000) + 500)
        cv2.setTrackbarPos("TY [-0.5,0.5]", self.control_window, int(self.ty * 1000) + 500)
        cv2.setTrackbarPos("TZ [-0.5,0.5]", self.control_window, int(self.tz * 1000) + 500)
        cv2.setTrackbarPos("Roll [-180°,180°]", self.control_window, int(math.degrees(self.roll) * 10) + 1800)
        cv2.setTrackbarPos("Pitch [-180°,180°]", self.control_window, int(math.degrees(self.pitch) * 10) + 1800)
        cv2.setTrackbarPos("Yaw [-180°,180°]", self.control_window, int(math.degrees(self.yaw) * 10) + 1800)

    def update_camera_parameters(self):
        # Build rotation matrix using current parameters
        Rx = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(self.roll), -math.sin(self.roll)],
            [0.0, math.sin(self.roll), math.cos(self.roll)]
        ], device=self.device)
        
        Ry = torch.tensor([
            [math.cos(self.pitch), 0.0, math.sin(self.pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(self.pitch), 0.0, math.cos(self.pitch)]
        ], device=self.device)
        
        Rz = torch.tensor([
            [math.cos(self.yaw), -math.sin(self.yaw), 0.0],
            [math.sin(self.yaw), math.cos(self.yaw), 0.0],
            [0.0, 0.0, 1.0]
        ], device=self.device)
        
        self.back_to_front_rotation = torch.matmul(torch.matmul(Rz, Ry), Rx)
        self.back_to_front_translation = torch.tensor([self.tx, self.ty, self.tz], device=self.device)
        
        if self.maps_initialized and not self.calibration_mode:
            self.maps_initialized = False
            self.get_logger().info("Parameters updated, remapping will occur on next image")

    def init_mapping(self, img_height: int, img_width: int):
        """Initialize mapping matrices for equirectangular projection."""
        if self.out_width is None or self.out_height is None:
            self.get_logger().error("Output dimensions (out_width, out_height) are not set. Cannot initialize mapping.")
            return

        self.get_logger().info(f"Initializing equirectangular projection: fusing two {img_width}x{img_height} fisheye images to {self.out_width}x{self.out_height}")
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.cx = img_width / 2 + self.cx_offset
        self.cy = img_height / 2 + self.cy_offset
        
        y, x = torch.meshgrid(
            torch.arange(self.out_height, dtype=torch.float32, device=self.device),
            torch.arange(self.out_width, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        longitude = (x / self.out_width) * 2 * math.pi - math.pi
        latitude = (y / self.out_height) * math.pi - math.pi / 2
        
        X = torch.cos(latitude) * torch.sin(longitude)
        Y = torch.sin(latitude)
        Z = torch.cos(latitude) * torch.cos(longitude)
        
        self.front_mask = (Z >= 0)
        self.back_mask = (Z < 0)
        
        self.front_mask_np = self.front_mask.cpu().numpy()
        self.back_mask_np = self.back_mask.cpu().numpy()
        
        r_front = torch.sqrt(X[self.front_mask]**2 + Y[self.front_mask]**2).clamp_(min=1e-6)
        theta_front = torch.atan2(r_front, torch.abs(Z[self.front_mask]))
        r_fisheye_front = 2 * theta_front / math.pi * (self.img_width / 2)
        
        self.front_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.front_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        
        self.front_map_x[self.front_mask] = self.cx + X[self.front_mask] / r_front * r_fisheye_front
        self.front_map_y[self.front_mask] = self.cy + Y[self.front_mask] / r_front * r_fisheye_front
        
        back_X_tensor = X[self.back_mask]
        back_Y_tensor = Y[self.back_mask]
        back_Z_tensor = Z[self.back_mask]
        
        back_points = torch.stack([back_X_tensor, back_Y_tensor, back_Z_tensor], dim=1)
        
        rotation = self.back_to_front_rotation.to(dtype=torch.float32)
        translation = self.back_to_front_translation.to(dtype=torch.float32)
        
        transformed_points = torch.matmul(back_points, rotation.transpose(0, 1)) + translation
        
        X_back = -transformed_points[:, 0]
        Y_back = transformed_points[:, 1]
        Z_back = transformed_points[:, 2]
        
        r_back = torch.sqrt(X_back**2 + Y_back**2).clamp_(min=1e-6)
        theta_back = torch.atan2(r_back, torch.abs(Z_back))
        r_fisheye_back = 2 * theta_back / math.pi * (self.img_width / 2)
        
        self.back_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.back_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)

        self.back_map_x[self.back_mask] = self.cx + X_back / r_back * r_fisheye_back
        self.back_map_y[self.back_mask] = self.cy + Y_back / r_back * r_fisheye_back

        self.front_map_x_np = self.front_map_x.cpu().numpy()
        self.front_map_y_np = self.front_map_y.cpu().numpy()
        self.back_map_x_np = self.back_map_x.cpu().numpy()
        self.back_map_y_np = self.back_map_y.cpu().numpy()
        
        if self.front_mask_np is not None:
            front_mask_uint8 = self.front_mask_np.astype(np.uint8)
            self.front_edge = cv2.dilate(front_mask_uint8, self.blend_kernel, iterations=2) - \
                              cv2.erode(front_mask_uint8, self.blend_kernel, iterations=2)
            self.front_distance = cv2.distanceTransform(front_mask_uint8, cv2.DIST_L2, 3)
            self.front_distance = np.clip(self.front_distance * 0.3, 0, 1)
        
        self.maps_initialized = True

        if self.use_cuda:
            try:
                if self.img_width is None or self.img_height is None: 
                    self.get_logger().error("Image dimensions (img_width, img_height) are None. Cannot initialize GPU grids.")
                    self.use_cuda = False
                else:
                    front_map_x_norm = 2.0 * (self.front_map_x / self.img_width) - 1.0
                    front_map_y_norm = 2.0 * (self.front_map_y / self.img_height) - 1.0
                    self.front_grid = torch.stack([front_map_x_norm, front_map_y_norm], dim=-1).unsqueeze(0)
                    
                    back_map_x_norm = 2.0 * (self.back_map_x / self.img_width) - 1.0
                    back_map_y_norm = 2.0 * (self.back_map_y / self.img_height) - 1.0
                    self.back_grid = torch.stack([back_map_x_norm, back_map_y_norm], dim=-1).unsqueeze(0)
                    
                    if self.front_mask is not None:
                        self.front_mask_gpu = self.front_mask.float().unsqueeze(0).unsqueeze(0)
                    if self.back_mask is not None:
                        self.back_mask_gpu = self.back_mask.float().unsqueeze(0).unsqueeze(0)
                    
                    # GPU edge and blend masks
                    if self.front_edge is not None:
                        self.edge_mask_gpu = torch.from_numpy(self.front_edge.astype(np.float32)).to(self.device, non_blocking=True).unsqueeze(0).unsqueeze(0)
                    if self.front_distance is not None:
                        self.blend_weight_gpu = torch.from_numpy(self.front_distance.astype(np.float32)).to(self.device, non_blocking=True).unsqueeze(0).unsqueeze(0)
                    
                    self.get_logger().info("GPU acceleration resources initialized successfully")

            except Exception as e:
                self.use_cuda = False
                self.get_logger().error(f"Error initializing GPU acceleration resources, falling back to CPU: {e}")
        
        self.get_logger().info(f"Mapping matrices initialization complete (GPU: {self.use_cuda})")

    def image_callback(self, dual_fisheye_msg: Image):
        """Process the dual fisheye image to create equirectangular image"""
        try:
            dual_fisheye_img = self.bridge.imgmsg_to_cv2(dual_fisheye_msg, "rgb8")
            
            img_height, img_width_full, _ = dual_fisheye_img.shape
            midpoint = img_width_full // 2
            front_img_full = dual_fisheye_img[:, midpoint:]
            back_img_full = dual_fisheye_img[:, :midpoint]

            front_img_full = cv2.rotate(front_img_full, cv2.ROTATE_90_COUNTERCLOCKWISE)
            back_img_full = cv2.rotate(back_img_full, cv2.ROTATE_90_CLOCKWISE)
            
            # Store original uncropped images (always update in calibration mode)
            if self.calibration_mode or self.original_front_img is None or self.original_front_img.shape != front_img_full.shape:
                self.original_front_img = front_img_full.copy()
            if self.calibration_mode or self.original_back_img is None or self.original_back_img.shape != back_img_full.shape:
                self.original_back_img = back_img_full.copy()

            # Crop images based on crop_size parameter
            current_crop_size = self.crop_size
            orig_h, orig_w = front_img_full.shape[:2]

            if orig_h != current_crop_size or orig_w != current_crop_size:
                y_start = (orig_h - current_crop_size) // 2
                x_start = (orig_w - current_crop_size) // 2
                
                if y_start >= 0 and x_start >= 0 and \
                   y_start + current_crop_size <= orig_h and x_start + current_crop_size <= orig_w:
                    front_img = front_img_full[y_start:y_start+current_crop_size, x_start:x_start+current_crop_size]
                    back_img = back_img_full[y_start:y_start+current_crop_size, x_start:x_start+current_crop_size]
                else:
                    # self.get_logger().warn(f"Cannot crop to {current_crop_size}x{current_crop_size}. Using uncropped images.")
                    front_img = front_img_full
                    back_img = back_img_full
            else:
                front_img = front_img_full
                back_img = back_img_full
            
            self.last_front_img = front_img.copy()
            self.last_back_img = back_img.copy()
            
            # Initialize mapping if needed (skip dimension check in calibration mode)
            if self.calibration_mode:
                # In calibration mode, only reinitialize when explicitly requested
                if not self.maps_initialized or self.params_changed:
                    self.init_mapping(front_img.shape[0], front_img.shape[1])
                    self.params_changed = False
            else:
                # Normal mode: reinitialize on dimension changes
                if not self.maps_initialized or self.params_changed or \
                   (self.img_height is not None and front_img.shape[0] != self.img_height) or \
                   (self.img_width is not None and front_img.shape[1] != self.img_width):
                    self.init_mapping(front_img.shape[0], front_img.shape[1])
                    self.params_changed = False
            
            start_time = self.get_clock().now()
            if self.use_cuda and self.maps_initialized:
                try:
                    equirect_img = self.create_equirectangular_gpu(front_img, back_img)
                except Exception as e:
                    self.get_logger().warn(f"GPU processing error: {e}, falling back to CPU")
                    self.use_cuda = False
                    equirect_img = self.create_equirectangular(front_img, back_img)
            else:
                if not self.maps_initialized:
                    self.get_logger().warn("Maps not initialized, attempting CPU processing with on-the-fly init.")
                equirect_img = self.create_equirectangular(front_img, back_img)
            
            equirect_msg = self.bridge.cv2_to_imgmsg(equirect_img, encoding="rgb8")
            equirect_msg.header = dual_fisheye_msg.header
            self.equirect_pub.publish(equirect_msg)
            
            process_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
            self.get_logger().debug(f"Processing time: {process_time:.3f} seconds (GPU: {self.use_cuda if self.maps_initialized else 'N/A'})")
            
            if self.calibration_mode:
                self.update_calibration_view()
            
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing images: {e}")

    def create_equirectangular(self, front_img: np.ndarray, back_img: np.ndarray) -> np.ndarray:
        """Create equirectangular image from front and back fisheye images using CPU."""
        if not self.maps_initialized or self.params_changed or \
           (self.img_height is not None and front_img.shape[0] != self.img_height) or \
           (self.img_width is not None and front_img.shape[1] != self.img_width):
            self.init_mapping(front_img.shape[0], front_img.shape[1])
            self.params_changed = False
        
        if not self.maps_initialized or self.front_map_x_np is None or self.front_map_y_np is None or \
           self.back_map_x_np is None or self.back_map_y_np is None or \
           self.front_mask_np is None or self.front_edge is None or \
           self.out_height is None or self.out_width is None:
            self.get_logger().error("CPU Mapping arrays or output dimensions not properly initialized. Returning black image.")
            h = self.out_height if self.out_height is not None else 1920
            w = self.out_width if self.out_width is not None else 3840
            return np.zeros((h, w, 3), dtype=np.uint8)
                
        front_result = cv2.remap(front_img, self.front_map_x_np, self.front_map_y_np, 
                                 cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        back_result = cv2.remap(back_img, self.back_map_x_np, self.back_map_y_np,
                                cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        
        equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
        
        non_edge_front = self.front_mask_np & (self.front_edge == 0)
        non_edge_back = ~self.front_mask_np & (self.front_edge == 0)

        equirect[non_edge_front] = front_result[non_edge_front]
        equirect[non_edge_back] = back_result[non_edge_back]
        
        if self.front_distance is not None:
            blend_region = (self.front_edge == 1)
            if np.any(blend_region):
                alpha = self.front_distance[blend_region][..., np.newaxis] 
                equirect[blend_region] = (alpha * front_result[blend_region].astype(np.float32) + 
                                        (1 - alpha) * back_result[blend_region].astype(np.float32)).astype(np.uint8)
        
        return equirect
    
    def create_equirectangular_gpu(self, front_img: np.ndarray, back_img: np.ndarray) -> np.ndarray:
        """Create equirectangular image using GPU acceleration."""
        if not self.use_cuda:
            self.get_logger().warn("GPU processing called but not enabled/initialized. Falling back to CPU.")
            return self.create_equirectangular(front_img, back_img)

        if not self.maps_initialized or self.params_changed or \
           (self.img_height is not None and front_img.shape[0] != self.img_height) or \
           (self.img_width is not None and front_img.shape[1] != self.img_width):
            self.init_mapping(front_img.shape[0], front_img.shape[1])
            self.params_changed = False
        
        if not self.maps_initialized or self.front_grid is None or self.back_grid is None or \
           self.front_mask_gpu is None or self.back_mask_gpu is None or \
           self.out_height is None or self.out_width is None:
            self.get_logger().error("GPU grids or masks not initialized. Falling back to CPU processing.")
            self.use_cuda = False 
            return self.create_equirectangular(front_img, back_img)

        front_tensor = torch.from_numpy(front_img).to(self.device, non_blocking=True).float().permute(2, 0, 1).unsqueeze(0)
        back_tensor = torch.from_numpy(back_img).to(self.device, non_blocking=True).float().permute(2, 0, 1).unsqueeze(0)
        
        front_remapped = F.grid_sample(front_tensor, self.front_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        back_remapped = F.grid_sample(back_tensor, self.back_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        if front_remapped.shape[2:] != (self.out_height, self.out_width):
            front_remapped = F.interpolate(front_remapped, size=(self.out_height, self.out_width), mode='bilinear', align_corners=True)
        if back_remapped.shape[2:] != (self.out_height, self.out_width):
            back_remapped = F.interpolate(back_remapped, size=(self.out_height, self.out_width), mode='bilinear', align_corners=True)
        
        output = front_remapped * self.front_mask_gpu + back_remapped * self.back_mask_gpu
        
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(output_np, 0, 255).astype(np.uint8)

    def setup_calibration_ui(self):
        """Set up UI for calibration mode"""
        # Window names
        self.window_name = "Equirectangular Calibration"
        self.control_window = "Calibration Controls"
        
        # Create windows
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.out_width // 2, self.out_height // 2)
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        
        # Create trackbars with initial values from ROS parameters
        cv2.createTrackbar("CX Offset [-100,100]", self.control_window, int(self.cx_offset) + 100, 200, self.update_cx)
        
        cv2.createTrackbar("CY Offset [-100,100]", self.control_window, int(self.cy_offset) + 100, 200, self.update_cy)
        
        cv2.createTrackbar("Crop Size", self.control_window, self.crop_size, 1920, self.update_crop)
        
        cv2.createTrackbar("TX [-0.5,0.5]", self.control_window, int(self.tx * 1000) + 500, 1000, self.update_tx)
        
        cv2.createTrackbar("TY [-0.5,0.5]", self.control_window, int(self.ty * 1000) + 500, 1000, self.update_ty)
        
        cv2.createTrackbar("TZ [-0.5,0.5]", self.control_window, int(self.tz * 1000) + 500, 1000, self.update_tz)
        
        cv2.createTrackbar("Roll [-180°,180°]", self.control_window, int(math.degrees(self.roll) * 10) + 1800, 3600, self.update_roll)
        
        cv2.createTrackbar("Pitch [-180°,180°]", self.control_window, int(math.degrees(self.pitch) * 10) + 1800, 3600, self.update_pitch)
        
        cv2.createTrackbar("Yaw [-180°,180°]", self.control_window, int(math.degrees(self.yaw) * 10) + 1800, 3600, self.update_yaw)

    # Trackbar update callbacks for calibration
    def update_cx(self, value):
        self.cx_offset = float(value - 100)

    def update_cy(self, value):
        self.cy_offset = float(value - 100)
        
    def update_crop(self, value: int):
        self.crop_size = value

    def update_tx(self, value):
        self.tx = float((value - 500) / 1000.0)

    def update_ty(self, value):
        self.ty = float((value - 500) / 1000.0)

    def update_tz(self, value):
        self.tz = float((value - 500) / 1000.0)

    def update_roll(self, value):
        self.roll = float(math.radians((value - 1800) / 10.0))

    def update_pitch(self, value):
        self.pitch = float(math.radians((value - 1800) / 10.0))

    def update_yaw(self, value):
        self.yaw = float(math.radians((value - 1800) / 10.0))
    
    def apply_parameters(self):
        """Apply all parameter changes and update the visualization"""
        # Re-crop images if crop size changed
        if self.original_front_img is not None and self.original_back_img is not None:
            orig_height, orig_width = self.original_front_img.shape[:2]
            y_start = (orig_height - self.crop_size) // 2
            x_start = (orig_width - self.crop_size) // 2
            
            if y_start >= 0 and x_start >= 0 and \
               y_start + self.crop_size <= orig_height and x_start + self.crop_size <= orig_width:
                self.last_front_img = self.original_front_img[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size].copy()
                self.last_back_img = self.original_back_img[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size].copy()
                self.get_logger().info(f"Re-cropped images to {self.crop_size}x{self.crop_size}")
            else:
                self.get_logger().warn(f"Cannot re-crop to {self.crop_size}. Using current images.")
        
        # Update all parameters in ROS
        self.set_parameters([
            Parameter('cx_offset', Parameter.Type.DOUBLE, self.cx_offset),
            Parameter('cy_offset', Parameter.Type.DOUBLE, self.cy_offset),
            Parameter('crop_size', Parameter.Type.INTEGER, self.crop_size),
            Parameter('translation', Parameter.Type.DOUBLE_ARRAY, [self.tx, self.ty, self.tz]),
            Parameter('rotation_deg', Parameter.Type.DOUBLE_ARRAY, [
                math.degrees(self.roll),
                math.degrees(self.pitch),
                math.degrees(self.yaw)
            ])
        ])
        
        # Force remapping
        self.update_camera_parameters()
        self.maps_initialized = False
        self.params_changed = True
        
        self.get_logger().info("Parameters applied successfully")
    
    def update_calibration_view(self):
        """Update the calibration view with current images and parameters"""
        if not hasattr(self, 'window_name') or self.last_front_img is None or self.last_back_img is None:
            return
        
        # Only process if parameters have been applied
        if self.params_changed:
            equirect_rgb = self.create_equirectangular(self.last_front_img, self.last_back_img)
            if equirect_rgb is not None:
                equirect_bgr = cv2.cvtColor(equirect_rgb, cv2.COLOR_RGB2BGR)
                self._cached_equirect = equirect_bgr
            else:
                # Fallback to cached or black image
                if hasattr(self, '_cached_equirect'):
                    equirect_bgr = self._cached_equirect
                else:
                    equirect_bgr = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
            self.params_changed = False
        else:
            # Use cached image if no changes
            if not hasattr(self, '_cached_equirect') or self._cached_equirect is None:
                equirect_rgb = self.create_equirectangular(self.last_front_img, self.last_back_img)
                if equirect_rgb is not None:
                    equirect_bgr = cv2.cvtColor(equirect_rgb, cv2.COLOR_RGB2BGR)
                    self._cached_equirect = equirect_bgr
                else:
                    equirect_bgr = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
            else:
                equirect_bgr = self._cached_equirect
        
        info_text = (
            f"cx: {self.crop_size/2 + self.cx_offset:.1f}, cy: {self.crop_size/2 + self.cy_offset:.1f} | "
            f"crop: {self.crop_size} | "
            f"t: [{self.tx:.3f}, {self.ty:.3f}, {self.tz:.3f}] | "
            f"r: [{math.degrees(self.roll):.1f}, {math.degrees(self.pitch):.1f}, {math.degrees(self.yaw):.1f}]"
        )
        
        cv2.putText(
            equirect_bgr, 
            info_text,
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Add instructions
        if equirect_bgr is not None:
            cv2.putText(
                equirect_bgr,
                "Press 'a' to Apply | 's' to Save | 'q' to Quit",
                (10, equirect_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        
        # Only show image if it's valid
        if equirect_bgr is not None and equirect_bgr.size > 0:
            cv2.imshow(self.window_name, equirect_bgr)
        
        key = cv2.waitKey(1)
        if key == ord('a'):
            self.apply_parameters()
            self._cached_equirect = None  # Clear cache to force update
        elif key == ord('s'):
            self.save_calibration()
        elif key == ord('q'):
            self.get_logger().info("Exiting calibration mode")
            cv2.destroyAllWindows()
            self.calibration_mode = False


def main(args=None):
    rclpy.init(args=args)
    
    # Check for calibration mode
    enable_calibration = '--calibrate' in sys.argv
    
    node = EquirectangularNode(enable_calibration=enable_calibration)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        if node.calibration_mode and hasattr(node, 'window_name'):
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()