#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import math # math might be used by the actual fisheyeImgConv implementation
from triton_bringup.omnicv import fisheyeImgConv

def process_video(input_video_path, output_folder_path):
    """
    Processes an equirectangular video to generate cubemap and individual face videos.

    Args:
        input_video_path (str): Path to the input MP4 video file.
        output_folder_path (str): Path to the folder where output videos will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if input_width == 0 or input_height == 0:
        print(f"Error: Video dimensions ({input_width}x{input_height}) are invalid for {input_video_path}")
        cap.release()
        return

    print(f"Input video: {input_video_path} ({input_width}x{input_height} @ {fps:.2f} FPS, {total_frames} frames)")

    # Calculate the side length for each cubemap face
    # This matches the logic from your ROS node: side = equirectangular_width / 4
    side = int(input_width / 4)
    if side == 0:
        print(f"Error: Calculated cubemap side is 0. Input width {input_width} might be too small.")
        cap.release()
        return

    cubemap_output_width = 4 * side
    cubemap_output_height = 3 * side

    print(f"Cubemap face side length: {side}px")
    print(f"Full cubemap dimensions: {cubemap_output_width}x{cubemap_output_height}px")

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4

    video_outputs = {
        "cubemap_full": cv2.VideoWriter(os.path.join(output_folder_path, 'cubemap_full.mp4'), fourcc, fps, (cubemap_output_width, cubemap_output_height)),
        "cubemap_front": cv2.VideoWriter(os.path.join(output_folder_path, 'cubemap_front.mp4'), fourcc, fps, (side, side)),
        "cubemap_back": cv2.VideoWriter(os.path.join(output_folder_path, 'cubemap_back.mp4'), fourcc, fps, (side, side)),
        "cubemap_left": cv2.VideoWriter(os.path.join(output_folder_path, 'cubemap_left.mp4'), fourcc, fps, (side, side)),
        "cubemap_right": cv2.VideoWriter(os.path.join(output_folder_path, 'cubemap_right.mp4'), fourcc, fps, (side, side)),
        "cubemap_down": cv2.VideoWriter(os.path.join(output_folder_path, 'cubemap_down.mp4'), fourcc, fps, (side, side)),
    }

    # Instantiate the image converter
    mapper = fisheyeImgConv()

    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break # End of video

        frame_count += 1
        if frame_count % 100 == 0 or frame_count == 1 or frame_count == total_frames : # Print progress
             print(f"Processing frame {frame_count}/{total_frames}...")

        # Convert frame from BGR (OpenCV default) to RGB, as fisheyeImgConv likely expects RGB
        # based on the 'rgb8' encoding in the original ROS node.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Convert equirectangular frame to cubemap
        # The 'dice=1' parameter is assumed from your original code structure.
        cubemap_rgb = mapper.equirect2cubemap(frame_rgb, side=side, dice=1)

        if cubemap_rgb is None or cubemap_rgb.shape != (cubemap_output_height, cubemap_output_width, 3):
            print(f"Error: equirect2cubemap did not return a valid cubemap image for frame {frame_count}.")
            print(f"Expected shape: ({cubemap_output_height}, {cubemap_output_width}, 3), Got: {cubemap_rgb.shape if cubemap_rgb is not None else 'None'}")
            print("Skipping frame and continuing. Check your 'fisheyeImgConv' implementation.")
            continue


        # Convert the full cubemap from RGB back to BGR for OpenCV VideoWriter
        cubemap_bgr = cv2.cvtColor(cubemap_rgb, cv2.COLOR_RGB2BGR)
        video_outputs["cubemap_full"].write(cubemap_bgr)

        # Extract faces from the RGB cubemap (as per your original script's slicing)
        # Note: Slicing numpy arrays creates views, not copies, by default.
        # This is efficient.
        front_rgb = cubemap_rgb[side:2*side, side:2*side]
        back_rgb = cubemap_rgb[side:2*side, 3*side:4*side]
        left_rgb = cubemap_rgb[side:2*side, 0:side]
        right_rgb = cubemap_rgb[side:2*side, 2*side:3*side]
        down_rgb = cubemap_rgb[2*side:3*side, side:2*side]

        # Write faces (after converting them to BGR)
        video_outputs["cubemap_front"].write(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
        video_outputs["cubemap_back"].write(cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR))
        video_outputs["cubemap_left"].write(cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
        video_outputs["cubemap_right"].write(cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
        video_outputs["cubemap_down"].write(cv2.cvtColor(down_rgb, cv2.COLOR_RGB2BGR))

    # Release resources
    print("Processing complete. Releasing video resources...")
    cap.release()
    for writer in video_outputs.values():
        writer.release()

    print(f"Output videos saved in: {output_folder_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert an equirectangular MP4 video to cubemap and individual face videos.")
    parser.add_argument("--input", required=True, help="Path to the input equirectangular MP4 video file.")
    parser.add_argument("--output", required=True, help="Path to the folder where output videos will be saved.")
    args = parser.parse_args()

    process_video(args.input, args.output)

if __name__ == '__main__':
    main()