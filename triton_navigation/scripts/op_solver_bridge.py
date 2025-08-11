#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
import json
import os
from typing import Optional, List, Tuple

class OpSolverBridge(Node):
    def __init__(self):
        super().__init__('op_solver_bridge')
        
        # Parameters
        self.declare_parameter('cost_limit', 10000.0)
        self.declare_parameter('oplib_file', 'config/problem.oplib')
        self.declare_parameter('solution_file', 'config/stats.json')
        
        # Get package share directory for relative paths
        from ament_index_python.packages import get_package_share_directory
        self.package_dir = get_package_share_directory('triton_navigation')
        
        # Subscriber for markers
        self.marker_sub = self.create_subscription(
            MarkerArray,
            '/combined_markers',
            self.marker_callback,
            10)
            
        # Publisher for waypoints
        self.waypoint_pub = self.create_publisher(PoseArray, '/waypoints', 10)
        
        # Publisher for orienteering path
        self.path_pub = self.create_publisher(Path, '/orienteering_path', 10)
        
        # Services
        self.export_service = self.create_service(
            Trigger,
            'export_oplib',
            self.export_oplib_callback)
            
        self.create_waypoints_service = self.create_service(
            Trigger,
            'create_waypoints',
            self.create_waypoints_callback)
            
        self.last_markers: Optional[MarkerArray] = None
        self.sphere_indices: List[int] = []  # Track indices of sphere markers
        self.get_logger().info('Op Solver Bridge initialized')
        
    def marker_callback(self, msg: MarkerArray) -> None:
        self.last_markers = msg
        
    def get_file_path(self, param_name: str) -> str:
        """Get absolute file path from parameter, handling relative paths."""
        file_path = str(self.get_parameter(param_name).value)
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.package_dir, file_path)
        return file_path
        
    def export_oplib_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self.last_markers or len(self.last_markers.markers) == 0:
            response.success = False
            response.message = "No markers available"
            return response
            
        try:
            # Get OPLIB file path
            oplib_file = self.get_file_path('oplib_file')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(oplib_file), exist_ok=True)
            
            # Write OPLIB file
            oplib_content = self.generate_oplib_content(self.last_markers)
            
            with open(oplib_file, 'w') as f:
                f.write(oplib_content)
                
            self.get_logger().info(f"OPLIB file written to: {oplib_file}")
            
            response.success = True
            response.message = f"OPLIB file exported to {oplib_file}\n\n{oplib_content}"
            
        except Exception as e:
            response.success = False
            response.message = f"Error exporting OPLIB: {str(e)}"
            self.get_logger().error(response.message)
            
        return response
        
    def generate_oplib_content(self, markers: MarkerArray) -> str:
        """Generate OPLIB file content from markers."""
        # Filter out cylinder markers (obstacles)
        self.sphere_indices = []
        sphere_markers = []
        cylinder_count = 0
        
        for i, marker in enumerate(markers.markers):
            if marker.type == Marker.SPHERE:
                self.sphere_indices.append(i)
                sphere_markers.append(marker)
            elif marker.type == Marker.CYLINDER:
                cylinder_count += 1
        
        self.get_logger().info(f"Processing {len(sphere_markers)} spheres (debris), ignoring {cylinder_count} cylinders (obstacles)")
        
        # Scale to integers
        scale = 1000.0
        cost_limit = int(float(self.get_parameter('cost_limit').value) * scale)
        
        # Build OPLIB content
        lines = []
        lines.append("NAME: marker_problem")
        lines.append("TYPE: OP")
        lines.append("COMMENT: ROS2 marker-based problem with depot at origin")
        lines.append(f"DIMENSION: {len(sphere_markers) + 1}")  # +1 for depot
        lines.append(f"COST_LIMIT : {cost_limit}")
        lines.append("EDGE_WEIGHT_TYPE : EUC_2D")
        lines.append("NODE_COORD_SECTION")
        
        # Node 1 is the depot at origin (0, 0)
        lines.append("1 0 0")
        
        # Add sphere markers as nodes 2, 3, ...
        for i, marker in enumerate(sphere_markers):
            x = int(round(marker.pose.position.x * scale))
            y = int(round(marker.pose.position.y * scale))
            lines.append(f"{i+2} {x} {y}")
            
        lines.append("NODE_SCORE_SECTION")
        # Depot has score 0
        lines.append("1 0")
        # All debris markers have score 1
        for i in range(len(sphere_markers)):
            lines.append(f"{i+2} 1")
            
        lines.append("DEPOT_SECTION")
        lines.append("1")
        lines.append("-1")
        lines.append("EOF")
        
        return '\n'.join(lines)
        
    def create_waypoints_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            # Get solution file path
            solution_file = self.get_file_path('solution_file')
            
            # Check if file exists
            if not os.path.exists(solution_file):
                response.success = False
                response.message = f"Solution file not found: {solution_file}"
                return response
                
            # Read solution JSON
            with open(solution_file, 'r') as f:
                content = f.read()
                
            # Parse multiple JSON objects (one per line)
            solutions = []
            for line in content.strip().split('\n'):
                if line.strip():
                    solutions.append(json.loads(line))
                    
            if not solutions:
                response.success = False
                response.message = "No solutions found in file"
                return response
                
            # Use the last solution (most optimized)
            solution_data = solutions[-1]
            
            # Extract cycle from solution
            cycle = solution_data.get('sol', {}).get('cycle', [])
            if not cycle:
                response.success = False
                response.message = "No cycle found in solution"
                return response
                
            # Remove depot visits (node 1) and adjust indices
            waypoint_indices = []
            for node in cycle:
                if node > 1:  # Skip depot (node 1)
                    waypoint_indices.append(node - 2)  # Convert to 0-based marker index
                    
            self.get_logger().info(f"Solution cycle: {cycle}")
            self.get_logger().info(f"Waypoint indices (markers only): {waypoint_indices}")
            
            # Create and publish PoseArray
            pose_array = self.create_pose_array(waypoint_indices)
            self.waypoint_pub.publish(pose_array)
            
            # Create and publish Path
            path = self.create_path(waypoint_indices)
            self.path_pub.publish(path)
            
            # Get solution cost
            cost = solution_data.get('sol', {}).get('cap', 0)
            cost_meters = cost / 1000.0  # Convert back to meters
            
            response.success = True
            response.message = f"Created {len(waypoint_indices)} waypoints from solution. Total path cost: {cost_meters:.2f} meters"
            self.get_logger().info(response.message)
            
        except json.JSONDecodeError as e:
            response.success = False
            response.message = f"Error parsing JSON: {str(e)}"
            self.get_logger().error(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Error creating waypoints: {str(e)}"
            self.get_logger().error(response.message)
            
        return response
        
    def create_pose_array(self, waypoint_indices: List[int]) -> PoseArray:
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        
        if self.last_markers is None or not self.sphere_indices:
            self.get_logger().warning("No markers available for creating pose array")
            return pose_array
            
        for node_idx in waypoint_indices:
            # Map solution index to original marker index
            if node_idx < len(self.sphere_indices):
                original_idx = self.sphere_indices[node_idx]
                if original_idx < len(self.last_markers.markers):
                    marker = self.last_markers.markers[original_idx]
                    pose = Pose()
                    pose.position = marker.pose.position
                    pose.orientation.x = 0.0
                    pose.orientation.y = 0.0
                    pose.orientation.z = 0.0
                    pose.orientation.w = 1.0
                    pose_array.poses.append(pose)
                
        return pose_array
    
    def create_path(self, waypoint_indices: List[int]) -> Path:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        
        if self.last_markers is None or not self.sphere_indices:
            self.get_logger().warning("No markers available for creating path")
            return path
        
        # Start from origin (depot)
        origin_pose = PoseStamped()
        origin_pose.header = path.header
        origin_pose.pose.position.x = 0.0
        origin_pose.pose.position.y = 0.0
        origin_pose.pose.position.z = 0.0
        origin_pose.pose.orientation.x = 0.0
        origin_pose.pose.orientation.y = 0.0
        origin_pose.pose.orientation.z = 0.0
        origin_pose.pose.orientation.w = 1.0
        path.poses.append(origin_pose)
        
        # Add all waypoints in the cycle order
        for node_idx in waypoint_indices:
            # Map solution index to original marker index
            if node_idx < len(self.sphere_indices):
                original_idx = self.sphere_indices[node_idx]
                if original_idx < len(self.last_markers.markers):
                    marker = self.last_markers.markers[original_idx]
                    pose_stamped = PoseStamped()
                    pose_stamped.header = path.header
                    pose_stamped.pose.position = marker.pose.position
                    pose_stamped.pose.orientation.x = 0.0
                    pose_stamped.pose.orientation.y = 0.0
                    pose_stamped.pose.orientation.z = 0.0
                    pose_stamped.pose.orientation.w = 1.0
                    path.poses.append(pose_stamped)
        
        # Return to origin (depot)
        path.poses.append(origin_pose)
        
        self.get_logger().info(f"Created path with {len(path.poses)} poses (including depot at start and end)")
        return path

def main(args=None):
    rclpy.init(args=args)
    node = OpSolverBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()