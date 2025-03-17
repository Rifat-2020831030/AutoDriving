

import numpy as np
import yaml

def calculate_steering_angle(lane_offset, lookahead=100):
    theta_rad = np.arctan(lane_offset / lookahead)  # Compute angle in radians
    theta_deg = np.degrees(theta_rad)  # Convert to degrees
    theta_deg = theta_deg 
    return theta_deg  # Clamp to [-21, 21]

def get_lane_center_offset(mask, bottom_fraction=0.4):
    h, w = mask.shape
    start_row = int(h * (1 - bottom_fraction))
    roi = mask[start_row:h, :]  # Bottom part of the image

    # Find all lane pixels in the ROI
    lane_rows, lane_cols = np.where(roi > 0)
    if len(lane_cols) == 0:
        return None  # No lane detected

    image_center = w // 2

    # Separate lane pixels by side of the image center
    left_lane_cols = lane_cols[lane_cols < image_center]
    right_lane_cols = lane_cols[lane_cols >= image_center]

    # If lane is visible on both sides, compute average lane center
    if len(left_lane_cols) > 0 and len(right_lane_cols) > 0:
        lane_center = np.mean(lane_cols)
    # If no lane detected on the left side, assume default lane width offset (35cm) to the left
    elif len(left_lane_cols) == 0 and len(right_lane_cols) > 0:
        lane_center = image_center - 35
    # If no lane detected on the right side, assume default lane width offset (35cm) to the right
    elif len(right_lane_cols) == 0 and len(left_lane_cols) > 0:
        lane_center = image_center + 35
    else:
        return None

    offset = lane_center - image_center  # Positive = right offset, Negative = left offset
    # hanlde the case when the offset is too small
    if abs(offset) < 10:
        offset = 0
    return offset


def get_lane_center_2(lane_points, bottom_fraction=0.4, fixed_center_x=640, save_path="lane_center.yaml"):
    if len(lane_points) == 0:
        return None, None, None  # No lane detected

    # Find the Y-coordinate of the lowest (closest) part of the image
    max_y = np.max(lane_points[:, 0])

    # Only consider lane points in the lower part of the image
    y_threshold = max_y - (bottom_fraction * max_y)
    closest_points = lane_points[lane_points[:, 0] >= y_threshold]

    if len(closest_points) == 0:
        return None, None, None  # No valid closest points

    # Compute the lane center among the closest points
    avg_x = np.mean(closest_points[:, 1])  # Average X-coordinate
    avg_y = np.max(closest_points[:, 0])     # Bottom-most Y (closest to car)

    # Compute steering deviation from fixed image center (assumed fixed_center_x)
    deviation = avg_x - fixed_center_x
    angle = (deviation / fixed_center_x)   # Scale deviation to full angle range (approx.)

    # Save the lane center and steering angle to YAML
    # lane_center_info = {
    #     "center_x": int(avg_x),
    #     "center_y": int(avg_y),
    #     "angle": int(angle)
    # }
    # with open(save_path, "w") as file:
    #     yaml.dump(lane_center_info, file, default_flow_style=False)

    return int(deviation)