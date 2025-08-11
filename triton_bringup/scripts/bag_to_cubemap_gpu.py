#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time

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

# Try to import CUDA acceleration
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CuPy not installed. GPU acceleration disabled. Install with: pip install cupy-cuda11x")

class BagToCubemapGPUNode(Node):
    def __init__(self):
        super().__init__('bag_to_cubemap_gpu_node')
        
        # Declare parameters
        self.declare_parameter('bag_file', '/home/triton/Desktop/bag/session3/rosbag2_2025_06_24-18_32_54/rosbag2_2025_06_24-18_32_54_0.db3')
        self.declare_parameter('output_folder', '/home/triton/Desktop/bag/session3_processed/run1')
        self.declare_parameter('frame_skip', 40.0)
        self.declare_parameter('use_gpu', False)  # Default to False to avoid GPU errors
        self.declare_parameter('num_threads', 4)
        self.declare_parameter('buffer_size', 10)
        
        # Get parameters
        self.bag_file = self.get_parameter('bag_file').value
        self.output_folder = self.get_parameter('output_folder').value
        self.frame_skip = self.get_parameter('frame_skip').value
        self.use_gpu = self.get_parameter('use_gpu').value and CUDA_AVAILABLE
        self.num_threads = self.get_parameter('num_threads').value
        self.buffer_size = self.get_parameter('buffer_size').value
        
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
        
        # Threading queues
        self.decode_queue = queue.Queue(maxsize=self.buffer_size)
        self.process_queue = queue.Queue(maxsize=self.buffer_size)
        self.save_queue = queue.Queue(maxsize=self.buffer_size * 4)  # 4 images per frame
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
        
        # GPU memory pool for CuPy
        if self.use_gpu:
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()
        
        self.get_logger().info(f"Initialized BagToCubemapGPU node")
        self.get_logger().info(f"  Bag file: {self.bag_file}")
        self.get_logger().info(f"  Output folder: {self.output_folder}")
        self.get_logger().info(f"  Frame skip: {self.frame_skip} (processing every {int(self.frame_skip) + 1} frame{'s' if self.frame_skip > 0 else ''})")
        self.get_logger().info(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        self.get_logger().info(f"  Worker threads: {self.num_threads}")
    
    def create_output_directories(self):
        """Create output directory"""
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"Created output directory: {self.output_folder}")
    
    def init_h264_decoder(self):
        """Initialize H264 decoder using PyAV"""
        try:
            # Create codec context for H264
            self.codec = av.CodecContext.create('h264', 'r')
            # Enable multithreading for decoder
            self.codec.thread_type = 'AUTO'
            self.codec.thread_count = 0  # Auto-detect
            self.get_logger().info("H264 decoder initialized successfully with multithreading")
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
    
    def dual_fisheye_to_equirectangular_gpu(self, dual_fisheye_img):
        """GPU-accelerated dual fisheye to equirectangular conversion"""
        try:
            # Convert to GPU memory
            gpu_img = cp.asarray(dual_fisheye_img)
            
            # Split the dual fisheye image
            img_h, img_w = gpu_img.shape[:2]
            midpoint = img_w // 2
            
            front_img = gpu_img[:, midpoint:]
            back_img = gpu_img[:, :midpoint]
            
            # Rotate images using CuPy
            front_img = cp.rot90(front_img, k=3)  # 90 degrees counterclockwise
            back_img = cp.rot90(back_img, k=1)    # 90 degrees clockwise
            
            # Determine crop size from image height
            crop_size = img_h
            
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
                self.init_equirectangular_mapping_gpu()
            
            # Convert back to CPU for cv2.remap (until we implement GPU remap)
            front_cpu = cp.asnumpy(front_img)
            back_cpu = cp.asnumpy(back_img)
            
            # Apply remapping
            front_remapped = cv2.remap(front_cpu, self.front_map_x, self.front_map_y, 
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            back_remapped = cv2.remap(back_cpu, self.back_map_x, self.back_map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            # Combine using masks
            equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
            equirect[self.front_mask] = front_remapped[self.front_mask]
            equirect[~self.front_mask] = back_remapped[~self.front_mask]
            
            return equirect
        except Exception as e:
            # If GPU processing fails, fall back to CPU
            if not hasattr(self, '_gpu_warning_shown'):
                self.get_logger().warning(f"GPU processing failed: {e}. Falling back to CPU mode.")
                self._gpu_warning_shown = True
            self.use_gpu = False
            return self.dual_fisheye_to_equirectangular(dual_fisheye_img)
    
    def dual_fisheye_to_equirectangular(self, dual_fisheye_img):
        """Convert dual fisheye image to equirectangular projection"""
        if self.use_gpu:
            return self.dual_fisheye_to_equirectangular_gpu(dual_fisheye_img)
        
        # Original CPU implementation
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
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        back_remapped = cv2.remap(back_img, self.back_map_x, self.back_map_y,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Combine using masks
        equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
        equirect[self.front_mask] = front_remapped[self.front_mask]
        equirect[~self.front_mask] = back_remapped[~self.front_mask]
        
        return equirect
    
    def init_equirectangular_mapping_gpu(self):
        """Initialize mapping matrices using GPU acceleration"""
        try:
            self.cx = self.crop_size / 2.0 + self.cx_offset
            self.cy = self.crop_size / 2.0 + self.cy_offset
            
            # Use CuPy for GPU computation
            x = cp.arange(self.out_width, dtype=cp.float32)
            y = cp.arange(self.out_height, dtype=cp.float32)
            x, y = cp.meshgrid(x, y)
        
            # Convert to spherical coordinates
            longitude = (x / self.out_width) * 2 * cp.pi - cp.pi
            latitude = (y / self.out_height) * cp.pi - cp.pi / 2
            
            # Convert to 3D coordinates
            X = cp.cos(latitude) * cp.sin(longitude)
            Y = cp.sin(latitude)
            Z = cp.cos(latitude) * cp.cos(longitude)
            
            # Create masks
            self.front_mask = cp.asnumpy(Z >= 0)
            
            # Build rotation matrix
            Rx = cp.array([[1.0, 0.0, 0.0],
                           [0.0, cp.cos(self.roll), -cp.sin(self.roll)],
                           [0.0, cp.sin(self.roll), cp.cos(self.roll)]])
            
            Ry = cp.array([[cp.cos(self.pitch), 0.0, cp.sin(self.pitch)],
                           [0.0, 1.0, 0.0],
                           [-cp.sin(self.pitch), 0.0, cp.cos(self.pitch)]])
            
            Rz = cp.array([[cp.cos(self.yaw), -cp.sin(self.yaw), 0.0],
                           [cp.sin(self.yaw), cp.cos(self.yaw), 0.0],
                           [0.0, 0.0, 1.0]])
            
            rotation = Rz @ Ry @ Rx
            translation = cp.array([self.tx, self.ty, self.tz])
            
            # Process on GPU and transfer to CPU for cv2.remap
            self.front_map_x, self.front_map_y, self.back_map_x, self.back_map_y = self._compute_maps_gpu(
                X, Y, Z, rotation, translation)
            
            self.equirect_initialized = True
            self.get_logger().info(f"GPU-accelerated equirectangular mapping initialized for {self.crop_size}x{self.crop_size} fisheye images")
        except Exception as e:
            # If GPU initialization fails, fall back to CPU
            self.get_logger().warning(f"GPU mapping initialization failed: {e}. Using CPU mode.")
            self.use_gpu = False
            self.init_equirectangular_mapping()
    
    def _compute_maps_gpu(self, X, Y, Z, rotation, translation):
        """Compute mapping arrays on GPU"""
        # Initialize mapping arrays
        front_map_x = cp.zeros((self.out_height, self.out_width), dtype=cp.float32)
        front_map_y = cp.zeros((self.out_height, self.out_width), dtype=cp.float32)
        back_map_x = cp.zeros((self.out_height, self.out_width), dtype=cp.float32)
        back_map_y = cp.zeros((self.out_height, self.out_width), dtype=cp.float32)
        
        # Process front hemisphere
        front_mask = Z >= 0
        X_front = X[front_mask]
        Y_front = Y[front_mask]
        Z_front = Z[front_mask]
        
        r_front = cp.sqrt(X_front**2 + Y_front**2)
        r_front = cp.maximum(r_front, 1e-6)
        theta_front = cp.arctan2(r_front, cp.abs(Z_front))
        r_fisheye_front = 2 * theta_front / cp.pi * (self.crop_size / 2.0)
        
        front_map_x[front_mask] = self.cx + X_front / r_front * r_fisheye_front
        front_map_y[front_mask] = self.cy + Y_front / r_front * r_fisheye_front
        
        # Process back hemisphere
        back_mask = ~front_mask
        points_back = cp.stack([X[back_mask], Y[back_mask], Z[back_mask]], axis=1)
        
        # Transform points
        transformed = (rotation @ points_back.T).T + translation
        
        X_back = -transformed[:, 0]
        Y_back = transformed[:, 1]
        Z_back = transformed[:, 2]
        
        r_back = cp.sqrt(X_back**2 + Y_back**2)
        r_back = cp.maximum(r_back, 1e-6)
        theta_back = cp.arctan2(r_back, cp.abs(Z_back))
        r_fisheye_back = 2 * theta_back / cp.pi * (self.crop_size / 2.0)
        
        back_map_x[back_mask] = self.cx + X_back / r_back * r_fisheye_back
        back_map_y[back_mask] = self.cy + Y_back / r_back * r_fisheye_back
        
        # Transfer to CPU memory
        return (cp.asnumpy(front_map_x), cp.asnumpy(front_map_y),
                cp.asnumpy(back_map_x), cp.asnumpy(back_map_y))
    
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
    
    def decode_worker(self, stop_event):
        """Worker thread for decoding H264 frames"""
        while not stop_event.is_set():
            try:
                msg, frame_num = self.decode_queue.get(timeout=0.1)
                if msg is None:  # Poison pill
                    break
                
                bgr_img = self.decode_compressed_image(msg)
                if bgr_img is not None:
                    self.process_queue.put((bgr_img, frame_num))
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Decode worker error: {e}")
    
    def process_worker(self, stop_event):
        """Worker thread for processing frames to cubemap"""
        while not stop_event.is_set():
            try:
                item = self.process_queue.get(timeout=0.1)
                if item is None:  # Poison pill
                    break
                
                bgr_img, frame_num = item
                
                # Convert to equirectangular
                equirect_img = self.dual_fisheye_to_equirectangular(bgr_img)
                
                # Convert to cubemap faces
                front, back, left, right = self.equirectangular_to_cubemap(equirect_img)
                
                # Add to save queue
                self.save_queue.put(('front', front, frame_num))
                self.save_queue.put(('back', back, frame_num))
                self.save_queue.put(('left', left, frame_num))
                self.save_queue.put(('right', right, frame_num))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Process worker error: {e}")
    
    def save_worker(self, stop_event):
        """Worker thread for saving images to disk"""
        while not stop_event.is_set():
            try:
                item = self.save_queue.get(timeout=0.1)
                if item is None:  # Poison pill
                    break
                
                face_name, image, frame_num = item
                filename = os.path.join(self.output_folder, f'{face_name}_frame_{frame_num:06d}.jpg')
                cv2.imwrite(filename, image)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Save worker error: {e}")
    
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
        
        # Start worker threads
        stop_event = threading.Event()
        
        # Start decode worker
        decode_thread = threading.Thread(target=self.decode_worker, args=(stop_event,))
        decode_thread.start()
        
        # Start process workers
        process_threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.process_worker, args=(stop_event,))
            t.start()
            process_threads.append(t)
        
        # Start save worker
        save_thread = threading.Thread(target=self.save_worker, args=(stop_event,))
        save_thread.start()
        
        # Process messages with progress bar
        start_time = time.time()
        with tqdm(total=message_count, desc="Processing frames") as pbar:
            while reader.has_next():
                topic_name, data, timestamp = reader.read_next()
                
                # Deserialize message
                msg_type = get_message(type_map[topic_name])
                msg = deserialize_message(data, msg_type)
                
                # Always decode compressed image to maintain H264 stream integrity
                self.decode_queue.put((msg, self.frame_count))
                
                # Update message count
                self.message_count += 1
                pbar.update(1)
                
                # Check if we should save this frame based on frame_skip
                should_save = (self.frame_skip == 0) or (self.decode_success_count % (int(self.frame_skip) + 1) == 0)
                
                if should_save:
                    self.frame_count += 1
        
        # Signal workers to stop
        self.decode_queue.put((None, None))  # Poison pill for decode worker
        for _ in range(self.num_threads):
            self.process_queue.put(None)  # Poison pills for process workers
        for _ in range(4):  # 4 images per frame
            self.save_queue.put(None)  # Poison pills for save worker
        
        # Wait for all workers to finish
        decode_thread.join()
        for t in process_threads:
            t.join()
        save_thread.join()
        
        # Clean up GPU memory if used
        if self.use_gpu:
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks()
        
        elapsed_time = time.time() - start_time
        fps = self.decode_success_count / elapsed_time if elapsed_time > 0 else 0
        
        self.get_logger().info(f"Processing complete!")
        self.get_logger().info(f"  Total messages in bag: {message_count}")
        self.get_logger().info(f"  Successfully decoded frames: {self.decode_success_count}")
        if self.frame_skip > 0:
            self.get_logger().info(f"  Frame skip: every {int(self.frame_skip) + 1} frames saved")
        self.get_logger().info(f"  Saved cubemap frames: {self.frame_count}")
        self.get_logger().info(f"  Processing time: {elapsed_time:.2f} seconds")
        self.get_logger().info(f"  Average FPS: {fps:.2f}")
        self.get_logger().info(f"  Output directory: {self.output_folder}")

def main(args=None):
    rclpy.init(args=args)
    
    node = BagToCubemapGPUNode()
    
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