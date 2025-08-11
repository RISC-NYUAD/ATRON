#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm

import rclpy
from rclpy.node import Node
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage
from triton_bringup.omnicv import fisheyeImgConv

try:
    import av
except ImportError:
    print("Error: PyAV not installed. Please install with: pip install av")
    sys.exit(1)

class BagToCubemapNode(Node):
    def __init__(self):
        super().__init__('bag_to_cubemap_node')
        
        # Declare parameters
        self.declare_parameter('bag_file', '/home/triton/Desktop/bag/session3/rosbag2_2025_06_24-18_32_54/rosbag2_2025_06_24-18_32_54_0.db3')
        self.declare_parameter('output_folder', '/home/triton/Desktop/bag/session3_processed/run1')
        self.declare_parameter('frame_skip', 0.0)
        
        # Get parameters
        self.bag_file = self.get_parameter('bag_file').value
        self.output_folder = self.get_parameter('output_folder').value
        self.frame_skip = self.get_parameter('frame_skip').value
        
        # Validate parameters
        if not self.bag_file:
            self.get_logger().error("bag_file parameter is required")
            sys.exit(1)
        
        if not self.output_folder:
            self.get_logger().error("output_folder parameter is required")
            sys.exit(1)
            
        if not os.path.exists(self.bag_file):
            self.get_logger().error(f"Bag file not found: {self.bag_file}")
            sys.exit(1)
        
        # Create output directories
        self.create_output_directories()
        
        # Initialize image converter for cubemap
        self.mapper = fisheyeImgConv()
        
        # Default parameters from equirectangular.cpp
        self.cx_offset = 0.0
        self.cy_offset = 0.0
        self.tx = 0.0
        self.ty = 0.0
        self.tz = -0.105
        self.roll = math.radians(-0.5)
        self.pitch = math.radians(0.0)
        self.yaw = math.radians(1.1)
        self.out_width = 1920
        self.out_height = 960
        
        # Initialize H264 decoder
        self.codec = None
        
        # Frame counter
        self.frame_count = 0  # Saved frames counter
        self.decode_success_count = 0  # Successfully decoded frames
        self.message_count = 0  # Total messages processed
        self.frames_to_save = 0  # Total frames that will be saved
        
        self.get_logger().info(f"Initialized BagToCubemap node")
        self.get_logger().info(f"  Bag file: {self.bag_file}")
        self.get_logger().info(f"  Output folder: {self.output_folder}")
        self.get_logger().info(f"  Frame skip: {self.frame_skip} (processing every {self.frame_skip + 1} frame{'s' if self.frame_skip > 0 else ''})")
    
    def create_output_directories(self):
        """Create output directory"""
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"Created output directory: {self.output_folder}")
    
    def init_h264_decoder(self):
        """Initialize H264 decoder using PyAV"""
        try:
            # Create codec context for H264
            self.codec = av.CodecContext.create('h264', 'r')
            self.get_logger().info("H264 decoder initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize H264 decoder: {e}")
            sys.exit(1)
    
    def decode_compressed_image(self, msg):
        """Decode H264 compressed image to BGR numpy array"""
        if msg.format != 'h264':
            self.get_logger().warn(f"Unsupported format: {msg.format}, expected h264")
            return None
        
        try:
            # Create packet from compressed data
            packet = av.Packet(bytes(msg.data))
            
            # Send packet to decoder
            frames = self.codec.decode(packet)
            
            # Get decoded frame
            for frame in frames:
                # Convert to numpy array (YUV420 to BGR)
                img = frame.to_ndarray(format='bgr24')
                self.decode_success_count += 1
                if self.decode_success_count == 1:
                    self.get_logger().info("Successfully decoded first frame")
                return img
                    
            return None
            
        except av.error.InvalidDataError:
            # Silently skip decoding errors for individual frames
            # This is common at the start of H264 streams until keyframe is found
            return None
        except Exception as e:
            # Log other unexpected errors only once
            if self.decode_success_count == 0 and not hasattr(self, '_logged_decode_error'):
                self.get_logger().debug(f"Waiting for keyframe... (error type: {type(e).__name__})")
                self._logged_decode_error = True
            return None
    
    def dual_fisheye_to_equirectangular(self, dual_fisheye_img):
        """Convert dual fisheye image to equirectangular projection"""
        # Split the dual fisheye image
        img_h, img_w = dual_fisheye_img.shape[:2]
        midpoint = img_w // 2
        
        front_img = dual_fisheye_img[:, midpoint:]
        back_img = dual_fisheye_img[:, :midpoint]
        
        # Rotate images as per equirectangular.cpp
        front_img = cv2.rotate(front_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        back_img = cv2.rotate(back_img, cv2.ROTATE_90_CLOCKWISE)
        
        # Determine crop size from image height
        crop_size = img_h  # For 1920x960 dual fisheye, crop_size = 960
        
        # Crop if necessary
        h, w = front_img.shape[:2]
        if h != crop_size or w != crop_size:
            y_start = (h - crop_size) // 2
            x_start = (w - crop_size) // 2
            
            if y_start >= 0 and x_start >= 0 and y_start + crop_size <= h and x_start + crop_size <= w:
                front_img = front_img[y_start:y_start+crop_size, x_start:x_start+crop_size]
                back_img = back_img[y_start:y_start+crop_size, x_start:x_start+crop_size]
        
        # Initialize mapping if needed
        if not hasattr(self, 'equirect_initialized') or self.crop_size != crop_size:
            self.crop_size = crop_size
            self.init_equirectangular_mapping()
        
        # Apply remapping
        front_remapped = cv2.remap(front_img, self.front_map_x, self.front_map_y, 
                                  cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        back_remapped = cv2.remap(back_img, self.back_map_x, self.back_map_y,
                                 cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        
        # Combine using masks
        equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
        equirect[self.front_mask] = front_remapped[self.front_mask]
        equirect[~self.front_mask] = back_remapped[~self.front_mask]
        
        return equirect
    
    def init_equirectangular_mapping(self):
        """Initialize mapping matrices for equirectangular projection"""
        self.cx = self.crop_size / 2.0 + self.cx_offset
        self.cy = self.crop_size / 2.0 + self.cy_offset
        
        # Create output coordinate grids
        x, y = np.meshgrid(np.arange(self.out_width), np.arange(self.out_height))
        
        # Convert to spherical coordinates
        longitude = (x / self.out_width) * 2 * np.pi - np.pi
        latitude = (y / self.out_height) * np.pi - np.pi / 2
        
        # Convert to 3D coordinates
        X = np.cos(latitude) * np.sin(longitude)
        Y = np.sin(latitude)
        Z = np.cos(latitude) * np.cos(longitude)
        
        # Create masks
        self.front_mask = Z >= 0
        
        # Build rotation matrix
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(self.roll), -np.sin(self.roll)],
                       [0.0, np.sin(self.roll), np.cos(self.roll)]])
        
        Ry = np.array([[np.cos(self.pitch), 0.0, np.sin(self.pitch)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(self.pitch), 0.0, np.cos(self.pitch)]])
        
        Rz = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0.0],
                       [np.sin(self.yaw), np.cos(self.yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        
        self.rotation = Rz @ Ry @ Rx
        self.translation = np.array([self.tx, self.ty, self.tz])
        
        # Initialize mapping arrays
        self.front_map_x = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        self.front_map_y = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        self.back_map_x = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        self.back_map_y = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        
        # Process front hemisphere
        front_indices = np.where(self.front_mask)
        X_front = X[front_indices]
        Y_front = Y[front_indices]
        Z_front = Z[front_indices]
        
        r_front = np.sqrt(X_front**2 + Y_front**2)
        r_front = np.maximum(r_front, 1e-6)
        theta_front = np.arctan2(r_front, np.abs(Z_front))
        r_fisheye_front = 2 * theta_front / np.pi * (self.crop_size / 2.0)
        
        self.front_map_x[front_indices] = self.cx + X_front / r_front * r_fisheye_front
        self.front_map_y[front_indices] = self.cy + Y_front / r_front * r_fisheye_front
        
        # Process back hemisphere
        back_indices = np.where(~self.front_mask)
        points_back = np.stack([X[back_indices], Y[back_indices], Z[back_indices]], axis=1)
        
        # Transform points
        transformed = (self.rotation @ points_back.T).T + self.translation
        
        X_back = -transformed[:, 0]
        Y_back = transformed[:, 1]
        Z_back = transformed[:, 2]
        
        r_back = np.sqrt(X_back**2 + Y_back**2)
        r_back = np.maximum(r_back, 1e-6)
        theta_back = np.arctan2(r_back, np.abs(Z_back))
        r_fisheye_back = 2 * theta_back / np.pi * (self.crop_size / 2.0)
        
        self.back_map_x[back_indices] = self.cx + X_back / r_back * r_fisheye_back
        self.back_map_y[back_indices] = self.cy + Y_back / r_back * r_fisheye_back
        
        self.equirect_initialized = True
        self.get_logger().info(f"Equirectangular mapping initialized for {self.crop_size}x{self.crop_size} fisheye images")
    
    def equirectangular_to_cubemap(self, equirect_img):
        """Convert equirectangular image to cubemap faces"""
        # Convert BGR to RGB for omnicv
        equirect_rgb = cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB)
        
        height, width = equirect_rgb.shape[:2]
        side = int(width / 4)
        
        # Generate cubemap using omnicv
        cubemap = self.mapper.equirect2cubemap(equirect_rgb, side=side, dice=True)
        
        # Extract faces (dice layout)
        front = cubemap[side:2*side, side:2*side]
        back = cubemap[side:2*side, 3*side:4*side]
        left = cubemap[side:2*side, 0:side]
        right = cubemap[side:2*side, 2*side:3*side]
        
        # Convert back to BGR for saving
        front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
        back_bgr = cv2.cvtColor(back, cv2.COLOR_RGB2BGR)
        left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        
        return front_bgr, back_bgr, left_bgr, right_bgr
    
    def process_bag(self):
        """Process the bag file and extract cubemap images"""
        self.init_h264_decoder()
        
        # Create bag reader
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_file,
            storage_id='sqlite3'
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader.open(storage_options, converter_options)
        
        # Get topic information
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
        
        # Check if our topic exists
        target_topic = '/dual_fisheye/image/compressed'
        if target_topic not in type_map:
            self.get_logger().error(f"Topic {target_topic} not found in bag file")
            self.get_logger().info(f"Available topics: {list(type_map.keys())}")
            return
        
        # Set filter for target topic
        storage_filter = rosbag2_py.StorageFilter(topics=[target_topic])
        reader.set_filter(storage_filter)
        
        # Count messages for progress bar
        self.get_logger().info("Counting messages...")
        message_count = 0
        while reader.has_next():
            reader.read_next()
            message_count += 1
        
        # Reset reader
        reader.reset_filter()
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        reader.set_filter(storage_filter)
        
        self.get_logger().info(f"Processing {message_count} compressed images...")
        
        # Process messages with progress bar
        with tqdm(total=message_count, desc="Processing frames") as pbar:
            while reader.has_next():
                topic_name, data, timestamp = reader.read_next()
                
                # Deserialize message
                msg_type = get_message(type_map[topic_name])
                msg = deserialize_message(data, msg_type)
                
                # Always decode compressed image to maintain H264 stream integrity
                bgr_img = self.decode_compressed_image(msg)
                
                # Update message count
                self.message_count += 1
                pbar.update(1)
                
                # Skip processing if no valid frame decoded
                if bgr_img is None:
                    continue
                
                # Check if we should save this frame based on frame_skip
                # frame_skip=0: save all frames
                # frame_skip=1: save every 2nd frame (0, 2, 4, ...)
                # frame_skip=n: save every (n+1)th frame
                should_save = (self.frame_skip == 0) or (self.decode_success_count % (int(self.frame_skip) + 1) == 0)
                
                if should_save:
                    # Convert to equirectangular
                    equirect_img = self.dual_fisheye_to_equirectangular(bgr_img)
                    
                    # Convert to cubemap faces
                    front, back, left, right = self.equirectangular_to_cubemap(equirect_img)
                    
                    # Save images with new naming convention
                    cv2.imwrite(os.path.join(self.output_folder, f'front_frame_{self.frame_count:06d}.jpg'), front)
                    cv2.imwrite(os.path.join(self.output_folder, f'back_frame_{self.frame_count:06d}.jpg'), back)
                    cv2.imwrite(os.path.join(self.output_folder, f'left_frame_{self.frame_count:06d}.jpg'), left)
                    cv2.imwrite(os.path.join(self.output_folder, f'right_frame_{self.frame_count:06d}.jpg'), right)
                    
                    self.frame_count += 1
        
        self.get_logger().info(f"Processing complete!")
        self.get_logger().info(f"  Total messages in bag: {message_count}")
        self.get_logger().info(f"  Successfully decoded frames: {self.decode_success_count}")
        if self.frame_skip > 0:
            self.get_logger().info(f"  Frame skip: every {int(self.frame_skip) + 1} frames saved")
        self.get_logger().info(f"  Saved cubemap frames: {self.frame_count}")
        self.get_logger().info(f"  Output directory: {self.output_folder}")

def main(args=None):
    rclpy.init(args=args)
    
    node = BagToCubemapNode()
    
    try:
        node.process_bag()
    except Exception as e:
        node.get_logger().error(f"Error processing bag: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()