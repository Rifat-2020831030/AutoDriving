import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import rospy
from scipy.interpolate import splprep, splev
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def compute_shortest_path(graph, start, end):
    # Extract node positions
    pos = {node: (float(graph.nodes[node]['x']), float(graph.nodes[node]['y'])) for node in graph.nodes}
    
    # Compute shortest path
    path_nodes = nx.shortest_path(graph, source=start, target=end, weight=None)
    
    # Extract path coordinates
    path_x = np.array([pos[node][0] for node in path_nodes])
    path_y = np.array([pos[node][1] for node in path_nodes])
    
    return path_x, path_y

def compute_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

def smooth_path(x, y):
    curvature = compute_curvature(x, y)
    
    # Define adaptive smoothing threshold
    threshold = np.where(curvature < 0.01, 0.35, 0.35)  # 35cm in straight, 350mm (35cm) in curves
    
    tck, u = splprep([x, y], s=threshold.mean())
    u_fine = np.linspace(0, 1, 100)
    smooth_x, smooth_y = splev(u_fine, tck)
    
    return smooth_x, smooth_y

def publish_waypoints(x, y):
    rospy.init_node('waypoint_publisher', anonymous=True)
    path_pub = rospy.Publisher('/waypoints', Path, queue_size=10)
    
    path_msg = Path()
    path_msg.header.frame_id = "map"
    
    for i in range(len(x)):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = x[i]
        pose.pose.position.y = y[i]
        path_msg.poses.append(pose)
    
    rospy.sleep(1)  # Ensure publisher is ready
    path_pub.publish(path_msg)
    print("Waypoints published.")

# Load the GraphML file
graphml_path = "Competition_track_graph.graphml"
G = nx.read_graphml(graphml_path)

x, y = compute_shortest_path(G, '263', '229')
smooth_x, smooth_y = smooth_path(x, y)
publish_waypoints(smooth_x, smooth_y)

pos = {node: (float(G.nodes[node]['x']), float(G.nodes[node]['y'])) for node in G.nodes}
nx.draw(G, pos=pos, with_labels=True, node_size=100, font_size=8)
plt.plot(x, y, 'ro-', label='Shortest path')
plt.plot(smooth_x, smooth_y, 'bo-', label='Smooth path')
plt.legend()
plt.show()
