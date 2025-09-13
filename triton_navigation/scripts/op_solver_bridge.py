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
        # Clustering radius in meters (adjacent markers within this distance are grouped)
        self.declare_parameter('cluster_radius', 1.5)
        
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
            
            # Build clustered waypoints (adjacent markers within radius are grouped)
            clustered_points = self.cluster_waypoints(waypoint_indices)

            # Create and publish PoseArray from clustered points
            pose_array = self.create_pose_array_from_points(clustered_points)
            self.waypoint_pub.publish(pose_array)

            # Create and publish Path from clustered points
            path = self.create_path_from_points(clustered_points)
            self.path_pub.publish(path)
            
            # Get solution cost
            cost = solution_data.get('sol', {}).get('cap', 0)
            cost_meters = cost / 1000.0  # Convert back to meters
            
            response.success = True
            response.message = (
                f"Created {len(clustered_points)} clustered waypoints from {len(waypoint_indices)} markers. "
                f"Total path cost: {cost_meters:.2f} meters"
            )
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
        
    def create_pose_array_from_points(self, points: List[Tuple[float, float]]) -> PoseArray:
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'

        if not points:
            self.get_logger().warning("No points available for creating pose array")
            return pose_array

        for x, y in points:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        return pose_array

    def create_path_from_points(self, points: List[Tuple[float, float]]) -> Path:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        if not points:
            self.get_logger().warning("No points available for creating path")
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

        # Add clustered waypoints in order
        for x, y in points:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path.poses.append(pose_stamped)

        # Return to origin (depot)
        path.poses.append(origin_pose)

        self.get_logger().info(f"Created clustered path with {len(points)} waypoints (including depot at start and end)")
        return path

    def ensure_sphere_indices(self) -> None:
        """Ensure sphere_indices is populated from the latest markers."""
        if self.last_markers is None:
            return
        if self.sphere_indices:
            return
        self.sphere_indices = [i for i, m in enumerate(self.last_markers.markers) if m.type == Marker.SPHERE]

    def cluster_waypoints(self, waypoint_indices: List[int]) -> List[Tuple[float, float]]:
        """Cluster adjacent markers in the ordered list by proximity.

        Returns a list of representative (x, y) points for each cluster in order.
        Adjacent markers within `cluster_radius` meters are grouped together.
        The representative is the centroid of each cluster.
        """
        # Guard conditions
        if self.last_markers is None:
            self.get_logger().warning("No markers available for clustering")
            return []

        # Ensure sphere index mapping exists
        self.ensure_sphere_indices()

        # Extract ordered positions for waypoint indices
        ordered_points: List[Tuple[float, float]] = []
        for node_idx in waypoint_indices:
            if 0 <= node_idx < len(self.sphere_indices):
                original_idx = self.sphere_indices[node_idx]
                if 0 <= original_idx < len(self.last_markers.markers):
                    m = self.last_markers.markers[original_idx]
                    ordered_points.append((m.pose.position.x, m.pose.position.y))

        if not ordered_points:
            return []

        radius = float(self.get_parameter('cluster_radius').value)

        # Cluster consecutive points based on pairwise distance
        clusters: List[List[Tuple[float, float]]] = []
        current_cluster: List[Tuple[float, float]] = [ordered_points[0]]

        def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return (dx*dx + dy*dy) ** 0.5

        for i in range(1, len(ordered_points)):
            if dist(ordered_points[i], ordered_points[i-1]) <= radius:
                current_cluster.append(ordered_points[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [ordered_points[i]]
        clusters.append(current_cluster)

        # Compute centroids for clusters
        cluster_centroids: List[Tuple[float, float]] = []
        for cluster in clusters:
            sx = sum(p[0] for p in cluster)
            sy = sum(p[1] for p in cluster)
            n = max(1, len(cluster))
            cluster_centroids.append((sx / n, sy / n))

        self.get_logger().info(
            f"Clustering applied: {len(ordered_points)} markers -> {len(cluster_centroids)} clusters (radius={radius} m)"
        )

        return cluster_centroids

def main(args=None):
    rclpy.init(args=args)
    node = OpSolverBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
