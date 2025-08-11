#!/usr/bin/env python3

import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math
import argparse
import json
import os
from tqdm import tqdm # For progress bar

class FisheyeToEquirectangularConverter:
    def __init__(self, calibration_file='extrinsics.json', use_gpu=False, 
                 out_width=1920, out_height=960, default_crop_size=960):
        """
        Initializes the Fisheye to Equirectangular Converter.

        Args:
            calibration_file (str): Path to the JSON calibration file.
            use_gpu (bool): Whether to use GPU acceleration.
            out_width (int): Width of the output equirectangular video.
            out_height (int): Height of the output equirectangular video.
            default_crop_size (int): Default crop size for fisheye images if not in calibration.
        """
        print(f"Initializing converter...")
        print(f"  Calibration file: {calibration_file}")
        print(f"  Use GPU: {use_gpu}")
        print(f"  Output resolution: {out_width}x{out_height}")

        # Check for CUDA availability
        self.use_cuda = torch.cuda.is_available() and use_gpu
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  Using GPU acceleration: {self.use_cuda}")
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Flag to track if mapping is initialized
        self.maps_initialized = False
        
        # Store image dimensions (will be set from the first processed frame)
        self.img_height = None
        self.img_width = None # This will be the width of a single fisheye image after cropping
        
        # Default parameter values (can be overridden by calibration file)
        self.cx_offset = 0.0
        self.cy_offset = 0.0
        self.crop_size = default_crop_size 
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.00 # Default to a small non-zero tz if not specified
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.out_width = out_width
        self.out_height = out_height
        
        # Try to load calibration from file
        self.calibration_file = calibration_file
        if not self.load_calibration():
            print(f"Warning: Could not load calibration from {calibration_file}. Using default parameters.")
        
        # If crop_size was loaded from calibration, it overrides default_crop_size
        # If not, default_crop_size is used.
        # self.img_height and self.img_width for init_mapping will be self.crop_size

        # Initialize rotation matrix and translation vector based on loaded/default params
        self.update_camera_transforms()
        print("Converter initialized.")

    def load_calibration(self):
        """Load calibration parameters from JSON file"""
        if not os.path.isfile(self.calibration_file):
            print(f"Calibration file not found: {self.calibration_file}")
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                params = json.load(f)
            
            self.cx_offset = params.get('cx_offset', self.cx_offset)
            self.cy_offset = params.get('cy_offset', self.cy_offset)
            self.crop_size = params.get('crop_size', self.crop_size) # Overrides default if present
            
            translation = params.get('translation', [self.tx, self.ty, self.tz])
            self.tx, self.ty, self.tz = translation
            
            rotation_deg = params.get('rotation_deg', [math.degrees(self.roll), math.degrees(self.pitch), math.degrees(self.yaw)])
            self.roll = math.radians(rotation_deg[0])
            self.pitch = math.radians(rotation_deg[1])
            self.yaw = math.radians(rotation_deg[2])
            
            # Output resolution can also be in calibration file
            self.out_width = params.get('out_width', self.out_width)
            self.out_height = params.get('out_height', self.out_height)

            print(f"Loaded calibration parameters from {self.calibration_file}")
            print(f"  Crop size: {self.crop_size}")
            print(f"  Center offset: ({self.cx_offset}, {self.cy_offset})")
            print(f"  Translation: [{self.tx}, {self.ty}, {self.tz}]")
            print(f"  Rotation (deg): {rotation_deg}")
            print(f"  Output resolution (from calib or default): {self.out_width}x{self.out_height}")
            return True
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            return False

    def update_camera_transforms(self):
        """Rebuild the rotation matrix and translation vector."""
        Rx = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(self.roll), -math.sin(self.roll)],
            [0.0, math.sin(self.roll), math.cos(self.roll)]
        ], device=self.device, dtype=torch.float32)
        
        Ry = torch.tensor([
            [math.cos(self.pitch), 0.0, math.sin(self.pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(self.pitch), 0.0, math.cos(self.pitch)]
        ], device=self.device, dtype=torch.float32)
        
        Rz = torch.tensor([
            [math.cos(self.yaw), -math.sin(self.yaw), 0.0],
            [math.sin(self.yaw), math.cos(self.yaw), 0.0],
            [0.0, 0.0, 1.0]
        ], device=self.device, dtype=torch.float32)
        
        self.back_to_front_rotation = torch.matmul(torch.matmul(Rz, Ry), Rx)
        self.back_to_front_translation = torch.tensor([self.tx, self.ty, self.tz], device=self.device, dtype=torch.float32)
        
        # If maps were initialized, they are now stale
        self.maps_initialized = False 
        print("Camera transforms updated.")

    def init_mapping(self):
        """
        Initialize mapping matrices for equirectangular projection.
        This should be called after self.crop_size is finalized and
        self.img_height and self.img_width are set to self.crop_size.
        """
        if self.img_height is None or self.img_width is None:
            # This means process_frame hasn't set them yet based on actual fisheye size
            # For init_mapping, the dimensions of the *individual, cropped* fisheye image are used.
            self.img_height = self.crop_size
            self.img_width = self.crop_size
        
        print(f"Initializing mapping matrices for input {self.img_width}x{self.img_height} (cropped fisheye) to output {self.out_width}x{self.out_height} (equirect)")
        
        # Set cx and cy based on crop size and offset
        # These are coordinates within the *individual cropped* fisheye image
        self.cx = self.img_width / 2.0 + self.cx_offset
        self.cy = self.img_height / 2.0 + self.cy_offset
        
        y, x = torch.meshgrid(
            torch.arange(self.out_height, dtype=torch.float32, device=self.device),
            torch.arange(self.out_width, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        longitude = (x / self.out_width) * 2 * math.pi - math.pi
        latitude = (y / self.out_height) * math.pi - math.pi/2.0
        
        X = torch.cos(latitude) * torch.sin(longitude)
        Y = torch.sin(latitude)
        Z = torch.cos(latitude) * torch.cos(longitude)
        
        self.front_mask = (Z >= 0)
        # self.back_mask = (Z < 0) # Not strictly needed if using ~self.front_mask

        # Calculate mapping for front camera
        r_front = torch.sqrt(X[self.front_mask]**2 + Y[self.front_mask]**2)
        r_front = torch.clamp(r_front, min=1e-6) # Avoid division by zero
        theta_front = torch.atan2(r_front, torch.abs(Z[self.front_mask]))
        
        # Radius in fisheye image (pixels). self.img_width is the diameter of the cropped fisheye.
        r_fisheye_front = (2.0 * theta_front / math.pi) * (self.img_width / 2.0)
        
        self.front_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.front_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        
        self.front_map_x[self.front_mask] = self.cx + X[self.front_mask]/r_front * r_fisheye_front
        self.front_map_y[self.front_mask] = self.cy + Y[self.front_mask]/r_front * r_fisheye_front
        
        # Calculate mapping for back camera
        back_mask_condition = ~self.front_mask
        back_X_sphere = X[back_mask_condition]
        back_Y_sphere = Y[back_mask_condition]
        back_Z_sphere = Z[back_mask_condition]
        
        back_points_sphere = torch.stack([back_X_sphere, back_Y_sphere, back_Z_sphere], dim=1)
        
        rotation = self.back_to_front_rotation.to(torch.float32)
        translation = self.back_to_front_translation.to(torch.float32)
        
        transformed_points = torch.matmul(back_points_sphere, rotation.T) # R^T for point transformation
        transformed_points = transformed_points + translation # Apply translation
        
        # Coordinates in the front camera's frame, but looking "backwards"
        X_back_cam_frame = -transformed_points[:, 0] # Negate X for fisheye projection
        Y_back_cam_frame = transformed_points[:, 1]
        Z_back_cam_frame = transformed_points[:, 2] # This Z is in front cam's frame
        
        r_back = torch.sqrt(X_back_cam_frame**2 + Y_back_cam_frame**2)
        r_back = torch.clamp(r_back, min=1e-6)
        # Use Z_back_cam_frame to determine theta for the back fisheye lens (which is now effectively the front lens)
        theta_back = torch.atan2(r_back, torch.abs(Z_back_cam_frame)) 
        r_fisheye_back = (2.0 * theta_back / math.pi) * (self.img_width / 2.0)
        
        self.back_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.back_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)

        self.back_map_x[back_mask_condition] = self.cx + X_back_cam_frame/r_back * r_fisheye_back
        self.back_map_y[back_mask_condition] = self.cy + Y_back_cam_frame/r_back * r_fisheye_back
        
        # For CPU (cv2.remap)
        self.front_map_x_np = self.front_map_x.cpu().numpy()
        self.front_map_y_np = self.front_map_y.cpu().numpy()
        self.back_map_x_np = self.back_map_x.cpu().numpy()
        self.back_map_y_np = self.back_map_y.cpu().numpy()
        self.front_mask_np = self.front_mask.cpu().numpy()

        self.maps_initialized = True
        print("Mapping matrices initialized.")

        if self.use_cuda:
            try:
                # Normalize coordinates for grid_sample: [-1, 1]
                # Input image dimensions for normalization are self.img_width, self.img_height (cropped fisheye size)
                front_map_x_norm = 2.0 * (self.front_map_x / self.img_width) - 1.0
                front_map_y_norm = 2.0 * (self.front_map_y / self.img_height) - 1.0
                self.front_grid = torch.stack([front_map_x_norm, front_map_y_norm], dim=-1).unsqueeze(0) # (1, H_out, W_out, 2)
                
                back_map_x_norm = 2.0 * (self.back_map_x / self.img_width) - 1.0
                back_map_y_norm = 2.0 * (self.back_map_y / self.img_height) - 1.0
                self.back_grid = torch.stack([back_map_x_norm, back_map_y_norm], dim=-1).unsqueeze(0)
                
                self.front_mask_gpu = self.front_mask.float().unsqueeze(0).unsqueeze(0) # (1, 1, H_out, W_out) for broadcasting
                print("GPU acceleration grids initialized successfully.")
            except Exception as e:
                self.use_cuda = False # Fallback to CPU if GPU init fails
                self.device = torch.device('cpu')
                print(f"Error initializing GPU acceleration grids, falling back to CPU: {e}")
        
    def process_frame(self, dual_fisheye_img_bgr):
        """
        Processes a single dual fisheye frame to create an equirectangular image.

        Args:
            dual_fisheye_img_bgr (np.ndarray): The input dual fisheye image (BGR format from OpenCV).

        Returns:
            np.ndarray: The processed equirectangular image (BGR format).
        """
        try:
            # Convert BGR to RGB as the original script worked with RGB
            dual_fisheye_img_rgb = cv2.cvtColor(dual_fisheye_img_bgr, cv2.COLOR_BGR2RGB)

            # Split the dual fisheye image into front and back
            # Assumes the input is two images side-by-side
            img_h_full, img_w_full = dual_fisheye_img_rgb.shape[:2]
            midpoint = img_w_full // 2
            
            front_img_raw = dual_fisheye_img_rgb[:, midpoint:]
            back_img_raw = dual_fisheye_img_rgb[:, 0:midpoint]

            front_img_raw = cv2.rotate(front_img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
            back_img_raw = cv2.rotate(back_img_raw, cv2.ROTATE_90_CLOCKWISE)

            # Crop images to self.crop_size
            # The raw fisheye images might be larger than self.crop_size (e.g. 1280x1280 from camera)
            # We need to crop them to self.crop_size x self.crop_size for the mapping.
            
            h_raw, w_raw = front_img_raw.shape[:2] # Should be same for back_img_raw

            if h_raw != self.crop_size or w_raw != self.crop_size:
                y_start = (h_raw - self.crop_size) // 2
                x_start = (w_raw - self.crop_size) // 2
                
                if y_start < 0 or x_start < 0 or \
                   y_start + self.crop_size > h_raw or \
                   x_start + self.crop_size > w_raw:
                    print(f"Error: Cannot crop {w_raw}x{h_raw} image to {self.crop_size}x{self.crop_size}. Check crop_size and input video resolution.")
                    # Create a black image of the expected output size to avoid crashing video writer
                    return np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)


                front_img_cropped = front_img_raw[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size]
                back_img_cropped = back_img_raw[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size]
            else:
                front_img_cropped = front_img_raw
                back_img_cropped = back_img_raw
            
            # Initialize mapping if this is the first frame or if fisheye dimensions changed (should not happen for video)
            # The crucial dimensions for init_mapping are those of the *cropped* fisheye images.
            if not self.maps_initialized:
                self.img_height = front_img_cropped.shape[0] # Should be self.crop_size
                self.img_width = front_img_cropped.shape[1]  # Should be self.crop_size
                if self.img_height != self.crop_size or self.img_width != self.crop_size:
                     print(f"Warning: Cropped image dimensions ({self.img_width}x{self.img_height}) "
                           f"do not match target crop_size ({self.crop_size}). This might affect calibration.")
                self.init_mapping()
            
            if self.use_cuda:
                equirect_img_rgb = self.create_equirectangular_gpu(front_img_cropped, back_img_cropped)
            else:
                equirect_img_rgb = self.create_equirectangular_cpu(front_img_cropped, back_img_cropped)
            
            # Convert back to BGR for OpenCV VideoWriter
            equirect_img_bgr = cv2.cvtColor(equirect_img_rgb, cv2.COLOR_RGB2BGR)
            return equirect_img_bgr
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            # Return a black frame of the correct output size to avoid crashing video writer
            return np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
    
    def create_equirectangular_cpu(self, front_img, back_img):
        """Create equirectangular image using CPU (cv2.remap)"""
        # front_img and back_img are cropped RGB numpy arrays
        
        front_result = cv2.remap(front_img, self.front_map_x_np, self.front_map_y_np, 
                                 cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        back_result = cv2.remap(back_img, self.back_map_x_np, self.back_map_y_np,
                                cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        equirect_img = np.zeros((self.out_height, self.out_width, 3), dtype=front_img.dtype)
        
        # Combine using the front_mask
        # Ensure masks are broadcastable to image channels
        equirect_img[self.front_mask_np] = front_result[self.front_mask_np]
        equirect_img[~self.front_mask_np] = back_result[~self.front_mask_np]
        
        return equirect_img.astype(np.uint8) # Ensure uint8 for video writing
    
    def create_equirectangular_gpu(self, front_img, back_img):
        """Create equirectangular image using GPU acceleration (torch.nn.functional.grid_sample)"""
        # front_img and back_img are cropped RGB numpy arrays
        
        # Convert images to PyTorch tensors (H, W, C) -> (C, H, W) -> (N, C, H, W)
        front_tensor = torch.from_numpy(front_img).to(self.device).float().permute(2, 0, 1).unsqueeze(0)
        back_tensor = torch.from_numpy(back_img).to(self.device).float().permute(2, 0, 1).unsqueeze(0)
        
        # grid_sample expects input (N, C, H_in, W_in) and grid (N, H_out, W_out, 2)
        # Output is (N, C, H_out, W_out)
        front_remapped = F.grid_sample(
            front_tensor, 
            self.front_grid, # Grid is for output dimensions
            mode='bilinear', padding_mode='zeros', align_corners=True # align_corners=True is often better
        )
        
        back_remapped = F.grid_sample(
            back_tensor, 
            self.back_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        # Combine using the GPU mask
        # self.front_mask_gpu is (1, 1, H_out, W_out)
        output_tensor = front_remapped * self.front_mask_gpu + back_remapped * (1.0 - self.front_mask_gpu)
        
        # Convert back to numpy array (N, C, H_out, W_out) -> (C, H_out, W_out) -> (H_out, W_out, C)
        output_np_rgb = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return np.clip(output_np_rgb, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Convert dual fisheye MP4 to equirectangular MP4.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input MP4 file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output MP4 file.')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to calibration JSON file.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration.')
    parser.add_argument('--out_width', type=int, default=3840, help='Output video width.')
    parser.add_argument('--out_height', type=int, default=1920, help='Output video height.')
    parser.add_argument('--crop_size', type=int, default=1920, help='Target size (width and height) for individual fisheye images after cropping. Should match calibration expectations.')
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    converter = FisheyeToEquirectangularConverter(
        calibration_file=args.calibration_file,
        use_gpu=args.gpu,
        out_width=args.out_width,
        out_height=args.out_height,
        default_crop_size=args.crop_size # Pass CLI crop_size as default
    )

    # Open video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {args.input}")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Input frame width/height are for the *dual* fisheye image
    # input_frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    # input_frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer
    # Output resolution is determined by converter.out_width and converter.out_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(args.output, fourcc, fps, (converter.out_width, converter.out_height))

    print(f"Processing video: {args.input}")
    print(f"  FPS: {fps}, Frame count: {frame_count}")
    print(f"  Outputting to: {args.output} at {converter.out_width}x{converter.out_height}")

    # Process each frame
    for i in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i+1}/{frame_count}. Ending early.")
            break
        
        processed_frame_bgr = converter.process_frame(frame_bgr)
        out.write(processed_frame_bgr)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows() # Just in case any debug windows were opened
    print(f"Video processing complete. Output saved to: {args.output}")

if __name__ == '__main__':
    main()
