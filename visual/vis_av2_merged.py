import pdb
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import cv2
from pathlib import Path
import random
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw
import copy
import imageio
from scipy.optimize import linear_sum_assignment
import matplotlib.transforms as transforms
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import time
from matplotlib.collections import LineCollection


def get_prev2curr_vectors(vecs=None, prev2curr_matrix=None):
    # transform prev vectors
    if len(vecs) > 0:
        vecs = np.stack(vecs, 0)
        vecs = torch.tensor(vecs)
        N, num_points, _ = vecs.shape
        denormed_vecs = torch.cat([
            vecs,
            vecs.new_zeros((N, num_points, 1)),  # z-axis
            vecs.new_ones((N, num_points, 1))  # 4-th dim
        ], dim=-1)  # (num_prop, num_pts, 4)

        transformed_vecs = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_vecs.double()).float()
        # vecs = (transformed_vecs[..., :2] - origin) / roi_size    # (num_prop, num_pts, 2)
    return transformed_vecs[..., :2]


def ska_points(map_points, horinzon):
    map_elements = []

    for ins in map_points:
        if ins is not None:
            points = np.array(ins)
            if horinzon:
                points[:, 0] = (100 - points[:, 0]) * 15 / 100
                points[:, 1] = (200 - points[:, 1]) * 15 / 100
            else:
                points[:, 0] = (points[:, 0] - 100) * 15 / 100
                points[:, 1] = (200 - points[:, 1]) * 15 / 100
            map_elements.append(points[:, ::-1])
    return map_elements


def plot_line(polyline, color, flip=False, linewidth=8):
    for i in range(len(polyline) - 1):
        if flip:
            plt.plot([polyline[i][0], polyline[i + 1][0]], [-polyline[i][1], -polyline[i + 1][1]],
                     c=color, linewidth=linewidth)
        else:
            plt.plot([polyline[i][0], polyline[i + 1][0]], [polyline[i][1], polyline[i + 1][1]],
                     c=color, linewidth=linewidth)


def plot_gt_unmerged(prev_gt_points, prev2curr_matrix):
    colors = {0: 'r', 1: 'b', 2: 'g', 3: 'g', 4: "c", }
    for item in prev_gt_points:
        pts = item['pts'][:, :2]
        pts[:, 1] = -pts[:, 1]
        pts = pts[np.newaxis, :]
        label = [item["type"]]
        # pdb.set_trace()
        prev2curr_gt_vectors = get_prev2curr_vectors(pts, prev2curr_matrix, )
        for i in range(len(prev2curr_gt_vectors)):
            ins = prev2curr_gt_vectors[i]
            for j in range(len(ins) - 1):
                plt.plot([ins[j][0], ins[j + 1][0]], [-ins[j][1], -ins[j + 1][1]], c=colors[int(label[i])])


def draw_polygons(vecs, roi_size, origin, cfg):
    results = []
    for poly_coords in vecs:
        mask = Image.new("L", size=(cfg["canvas_size"][0], cfg["canvas_size"][1]), color=0)
        coords = (poly_coords - origin) / roi_size * np.array(cfg["thickness"])
        coords = coords.numpy()
        vert_list = [(x, y) for x, y in coords]
        if not (coords[0] == coords[-1]).all():
            vert_list.append(vert_list[0])
        ImageDraw.Draw(mask).polygon(vert_list, outline=1, fill=1)
        result = np.flipud(np.array(mask))
        if result.sum() < 20:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            result = cv2.dilate(result, kernel, iterations=1)
        results.append(result)
    return results


def draw_polylines(vecs, roi_size, origin, cfg):
    results = []
    for line_coords in vecs:
        canvas = np.zeros((cfg["canvas_size"][1], cfg["canvas_size"][0]), dtype=np.uint8)  # thickness = 3
        coords = (line_coords - origin) / torch.tensor(roi_size) * torch.tensor(cfg["canvas_size"])
        # canvas = np.zeros((roi_size[1], roi_size[0]), dtype=np.uint8)
        # pdb.set_trace()
        coords = coords.numpy()
        cv2.polylines(canvas, np.int32([coords]), False, color=1, thickness=cfg["thickness"])
        result = np.flipud(canvas)  # 上下翻转
        if result.sum() < 20:  # 小于20个点 半条线
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 7x7椭圆形结构元素，它决定了膨胀的范围和形状
            result = cv2.dilate(result, kernel, iterations=1)  # 膨胀操作
        results.append(result)
    return results


def draw_instance_masks(vectors, roi_size, origin, cfg):
    masks = {}
    for label, vecs in vectors.items():
        # if label == 0:
        #     masks[label] = draw_polygons(vecs, roi_size, origin, cfg)
        # else:
        if len(vecs) == 0:
            masks[label] = []
            continue
        masks[label] = draw_polylines(vecs, roi_size, origin, cfg)
    return masks


def _mask_iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union


def find_matchings(src_masks, tgt_masks, thresh=0.1):
    matchings = {}
    for label, src_instances in src_masks.items():  # src_masks: prev
        tgt_instances = tgt_masks[label]
        if len(src_instances) == 0:
            if len(tgt_instances) == 0:
                matchings[label] = [[], []]
            else:
                label_matching_reverse = [-1 for _ in range(len(tgt_instances))]
                matchings[label] = [[], [label_matching_reverse]]
            continue
        else:
            if len(tgt_instances) == 0:
                label_matching = [-1 for _ in range(len(src_instances))]
                matchings[label] = [label_matching, []]
                continue

        cost = np.zeros([len(src_instances), len(tgt_instances)])
        for i, src_ins in enumerate(src_instances):
            for j, tgt_ins in enumerate(tgt_instances):
                iou = _mask_iou(src_ins, tgt_ins)
                cost[i, j] = -iou
        row_ind, col_ind = linear_sum_assignment(cost)

        label_matching = [-1 for _ in range(len(src_instances))]  # [-1, -1, -1, ....     # prev 对应 next
        label_matching_reverse = [-1 for _ in range(len(tgt_instances))]  # next 对应 prev
        for i, j in zip(row_ind, col_ind):
            if -cost[i, j] > thresh:  # 0.01
                label_matching[i] = j
                label_matching_reverse[j] = i

        matchings[label] = (label_matching, label_matching_reverse)
    return matchings


def match_frames(prev2curr_points_dict, next2curr_points_dict, roi_size, origin, cfg):
    # get vector data
    # prev_vectors = copy.deepcopy(prev2next_points_dict)
    # curr_vectors = copy.deepcopy(next_points_dict)
    # pdb.set_trace()
    # prev2curr_vectors = dict()
    # for label, vecs in prev_vectors.items():  # vecs [5, 100, 2]
    #     if len(vecs) > 0:
    #         normed_vecs = torch.tensor(vecs[..., :2] - origin) / roi_size  # (num_prop, num_pts, 2)
    #         normed_vecs = torch.clip(normed_vecs, min=0., max=1.)
    #         prev2curr_vectors[label] = normed_vecs * roi_size + origin
    #     # vectors = (prev_vectors - origin) / roi_size

    # convert to ego space for visualization
    # for label in prev2curr_vectors:
    #     if len(prev2curr_vectors[label]) > 0:
    #         prev2curr_vectors[label] = prev2curr_vectors[label] * roi_size + origin
    #     if len(curr_vectors[label]) > 0:
    #         curr_vecs = torch.tensor(np.stack(curr_vectors[label]))
    #         curr_vectors[label] = curr_vecs * roi_size + origin

    prev2curr_masks = draw_instance_masks(prev2curr_points_dict, roi_size, origin, cfg)
    next2curr_masks = draw_instance_masks(next2curr_points_dict, roi_size, origin, cfg)

    # plt.figure()
    # plt.subplot(4, 2, 1)
    # plt.imshow(prev2curr_masks["boundary"][0], cmap='gray')
    # plt.subplot(4, 2, 3)
    # plt.imshow(prev2curr_masks["boundary"][1], cmap='gray')
    # plt.subplot(4, 2, 5)
    # plt.imshow(prev2curr_masks["boundary"][2], cmap='gray')
    # plt.subplot(4, 2, 7)
    # plt.imshow(prev2curr_masks["boundary"][3], cmap='gray')
    # plt.subplot(4, 2, 2)
    # plt.imshow(next2curr_masks["boundary"][0], cmap='gray')
    # plt.subplot(4, 2, 4)
    # plt.imshow(next2curr_masks["boundary"][1], cmap='gray')
    # plt.subplot(4, 2, 6)
    # plt.imshow(next2curr_masks["boundary"][2], cmap='gray')
    # plt.show()
    # pdb.set_trace()
    prev2curr_matching = find_matchings(prev2curr_masks, next2curr_masks, thresh=0.01)

    return prev2curr_matching


def points_list_to_dict(points_list, pred_label):
    crossings, boundaries, dividers = [], [], []
    prev_points_dict = {}
    # pdb.set_trace()
    for m in range(len(pred_label)):
        if pred_label[m] == 1:
            dividers.append(points_list[m])
        elif pred_label[m] == 2:
            crossings.append(points_list[m])
        elif pred_label[m] == 3:
            boundaries.append(points_list[m])

    if len(crossings) > 0:
        prev_points_dict['crossing'] = np.stack(crossings)
    else:
        prev_points_dict['crossing'] = np.empty((0, 100, 2))
    if len(boundaries) > 0:
        prev_points_dict['boundary'] = np.stack(boundaries)
    else:
        prev_points_dict['boundary'] = np.empty((0, 100, 2))
    if len(dividers) > 0:
        prev_points_dict['divider'] = np.stack(dividers)
    else:
        prev_points_dict['divider'] = np.empty((0, 100, 2))
    return prev_points_dict


def get_matrix(input_dict, inverse):
    e2g_trans = torch.tensor(input_dict['ego2global_translation'], dtype=torch.float64)
    e2g_rot = torch.tensor(input_dict['ego2global_rotation'], dtype=torch.float64)
    matrix = torch.eye(4, dtype=torch.float64)
    if inverse:  # g2e_matrix
        matrix[:3, :3] = e2g_rot.T
        matrix[:3, 3] = -(e2g_rot.T @ e2g_trans)
    else:
        matrix[:3, :3] = e2g_rot
        matrix[:3, 3] = e2g_trans
    return matrix


def project_point_onto_line(point, line):
    """Project a point onto a line segment and return the projected point."""
    line_start, line_end = np.array(line.coords[0]), np.array(line.coords[1])
    line_vec = line_end - line_start
    point_vec = np.array(point.coords[0]) - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = np.clip(t, 0.0, 1.0)
    nearest = line_start + t * line_vec
    return Point(nearest)


def find_nearest_projection_on_polyline(point, polyline):
    """Find the nearest projected point of a point onto a polyline."""
    min_dist = float('inf')
    nearest_point = None
    for i in range(len(polyline.coords) - 1):
        segment = LineString(polyline.coords[i:i+2])
        proj_point = project_point_onto_line(point, segment)
        dist = point.distance(proj_point)
        if dist < min_dist:
            min_dist = dist
            nearest_point = proj_point
    return np.array(nearest_point.coords)


def find_and_sort_intersections(segmenet1, segment2):
    # Convert polylines to LineString objects       1：line, 2: intersection

    # Find the intersection between the two LineStrings  线和多边形
    intersection = segmenet1.intersection(segment2)

    # Prepare a list to store intersection points
    intersections = []

    # Check the type of intersection
    if "Point" in intersection.geom_type:
        # Single point or multiple points
        if intersection.geom_type == "MultiPoint":
            intersections.extend(list(intersection))
        else:
            intersections.append(intersection)
    elif "LineString" in intersection.geom_type:
        # In case of lines or multiline, get boundary points (start and end points of line segments)  只取首尾两个点
        if intersection.geom_type == "MultiLineString":
            for line in intersection:
                intersections.extend(list(line.boundary))
        else:
            intersections.extend(list(intersection.boundary))

    # Remove duplicates and ensure they are Point objects
    unique_intersections = [Point(coords) for coords in set(pt.coords[0] for pt in intersections)]  # 集合set自动去除重复项

    # Sort the intersection points by their distance along the first polyline
    sorted_intersections = sorted(unique_intersections, key=lambda pt: segmenet1.project(pt))  # 根据它们离 segment1 起点

    return sorted_intersections


def get_intersection_point_on_line(line, intersection):
    intersection_points = find_and_sort_intersections(LineString(line), intersection)
    if len(intersection_points) >= 2:
        line_intersect_start = intersection_points[0]
        line_intersect_end = intersection_points[-1]
    elif len(intersection_points) == 1:
        if intersection.contains(Point(line[0])):
            line_intersect_start = Point(line[0])
            line_intersect_end = intersection_points[0]
        elif intersection.contains(Point(line[-1])):
            line_intersect_start = Point(line[-1])
            line_intersect_end = intersection_points[0]
        else:
            return None, None
    else:
        return None, None
    return line_intersect_start, line_intersect_end


def merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end, car_trajectory):
    # get nearest point on line2 to line2_intersect_start
    line2_point_to_merge = []
    line2_intersect_start_dis = line2.project(line2_intersect_start)  # 0
    line2_intersect_end_dis = line2.project(line2_intersect_end)  # 2.1876873288601377
    for point in np.array(line2.coords):
        point_geom = Point(point)
        dis = line2.project(point_geom)
        if dis > line2_intersect_start_dis and dis < line2_intersect_end_dis:
            line2_point_to_merge.append(point)

    # merged the points
    merged_line2_points = []
    for point in line2_point_to_merge:
        # Use the `project` method to find the distance along the polyline to the closest point
        point_geom = Point(point)
        # Use the `interpolate` method to find the actual point on the polyline
        closest_point_on_line = find_nearest_projection_on_polyline(point_geom, line1)
        if len(closest_point_on_line) == 0:
            merged_line2_points.append(point)
        else:
            # merged_line2_points.append(((closest_point_on_line + point) / 2)[0])
            if car_trajectory is None:
                merged_line2_points.append(((closest_point_on_line + point) / 2)[0])
            else:
                # distance weight
                # dis2car = np.abs((car_trajectory[0] - point[0]) * (car_trajectory[1] - point[1]))
                # point_weight = np.array(min(max(1 - dis2car/50, 0.1), 0.8))
                # merged_line2_points.append((closest_point_on_line*(1-point_weight) + point*point_weight)[0])

                cams_Pose = [0, 45, 80, 153, 207, 280, 315]
                angle = np.arctan2((point[1] - car_trajectory[1]), (point[0] - car_trajectory[0])) / np.pi * 180
                if angle < 0:
                    angle = 360 + angle
                difference_min = min(np.abs(angle - np.array(cams_Pose)))
                angle_weight = np.array(min(0.8, 1 - difference_min / 38))
                merged_line2_points.append((closest_point_on_line * (1 - angle_weight) + point * angle_weight)[0])

    if len(merged_line2_points) == 0:
        merged_line2_points = np.array([]).reshape(0, 2)
    else:
        merged_line2_points = np.array(merged_line2_points)

    return merged_line2_points


def segment_line_based_on_merged_area(line, merged_points):
    if len(merged_points) == 0:
        return np.array(line.coords), np.array([]).reshape(0, 2)

    first_merged_point = merged_points[0]
    last_merged_point = merged_points[-1]

    start_dis = line.project(Point(first_merged_point))
    end_dis = line.project(Point(last_merged_point))

    start_segmenet = []
    for point in np.array(line.coords):
        point_geom = Point(point)
        if line.project(point_geom) < start_dis:
            start_segmenet.append(point)

    end_segmenet = []
    for point in np.array(line.coords):
        point_geom = Point(point)
        if line.project(point_geom) > end_dis:
            end_segmenet.append(point)

    if len(start_segmenet) == 0:
        start_segmenet = np.array([]).reshape(0, 2)
    else:
        start_segmenet = np.array(start_segmenet)

    if len(end_segmenet) == 0:
        end_segmenet = np.array([]).reshape(0, 2)
    else:
        end_segmenet = np.array(end_segmenet)

    return start_segmenet, end_segmenet


def get_bbox_size_for_points(points):
    if len(points) == 0:
        return 0, 0

    # Initialize min and max coordinates with the first point
    min_x, min_y = points[0]
    max_x, max_y = points[0]

    # Iterate through each point to update min and max coordinates
    for x, y in points[1:]:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return max_x - min_x, max_y - min_y


def get_longer_segmenent_to_merged_points(l1_segment, l2_segment, merged_line2_points, segment_type="start"):
    # remove points from segments if it's too close to merged_line2_points
    l1_segment_temp = []
    if len(merged_line2_points) > 1:
        merged_polyline = LineString(merged_line2_points)
        for point in l1_segment:
            if merged_polyline.distance(Point(point)) > 0.1:
                l1_segment_temp.append(point)
    elif len(merged_line2_points) == 1:
        for point in l1_segment:
            if Point(point).distance(Point(merged_line2_points[0])) > 0.1:
                l1_segment_temp.append(point)
    elif len(merged_line2_points) == 0:
        l1_segment_temp = l1_segment
    l1_segment = np.array(l1_segment_temp)

    l2_segmenet_temp = []
    if len(merged_line2_points) > 1:
        merged_polyline = LineString(merged_line2_points)
        for point in l2_segment:
            if merged_polyline.distance(Point(point)) > 0.1:
                l2_segmenet_temp.append(point)
    elif len(merged_line2_points) == 1:
        for point in l2_segment:
            if Point(point).distance(Point(merged_line2_points[0])) > 0.1:
                l2_segmenet_temp.append(point)
    elif len(merged_line2_points) == 0:
        l2_segmenet_temp = l2_segment
    l2_segment = np.array(l2_segmenet_temp)
    # ---------------------------------------
    if segment_type == "start":

        temp = l1_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[0])
        l1_start_box_size = get_bbox_size_for_points(temp)

        temp = l2_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[0])
        l2_start_box_size = get_bbox_size_for_points(temp)

        if l2_start_box_size[0] * l2_start_box_size[1] >= l1_start_box_size[0] * l1_start_box_size[1]:
            longer_segment = l2_segment
        else:
            longer_segment = l1_segment
    else:
        temp = l1_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[-1])
        l1_end_box_size = get_bbox_size_for_points(temp)

        temp = l2_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[-1])
        l2_end_box_size = get_bbox_size_for_points(temp)

        if l2_end_box_size[0] * l2_end_box_size[1] >= l1_end_box_size[0] * l1_end_box_size[1]:
            longer_segment = l2_segment
        else:
            longer_segment = l1_segment

    if len(longer_segment) == 0:
        longer_segment = np.array([]).reshape(0, 2)
    else:
        longer_segment = np.array(longer_segment)

    return longer_segment


def algin_l2_with_l1(line1, line2):
    if len(line1) > len(line2):
        l2_len = len(line2)
        line1_geom = LineString(line1)
        interval_length = line1_geom.length / (l2_len - 1)
        line1 = [np.array(line1_geom.interpolate(interval_length * i)) for i in range(l2_len)]

    elif len(line1) < len(line2):
        l1_len = len(line1)
        line2_geom = LineString(line2)
        interval_length = line2_geom.length / (l1_len - 1)
        line2 = [np.array(line2_geom.interpolate(interval_length * i)) for i in range(l1_len)]

    # make line1 and line2 same direction, pre_line.coords[0] shold be closer to line2.coords[0]
    line1_geom = LineString(line1)
    line2_flip = np.flip(line2, axis=0)

    line2_traj_len = 0
    for point_idx, point in enumerate(line2):
        line2_traj_len += np.linalg.norm(point - line1[point_idx])

    flip_line2_traj_len = 0
    for point_idx, point in enumerate(line2_flip):
        flip_line2_traj_len += np.linalg.norm(point - line1[point_idx])

    if abs(flip_line2_traj_len - line2_traj_len) < 3:
        # get the trajectory length
        line2_walk_len = 0
        for point in line2:
            point_geom = Point(point)
            proj_point = find_nearest_projection_on_polyline(point_geom, line1_geom)
            if len(proj_point) != 0:
                line2_walk_len += line1_geom.project(Point(proj_point[0]))

        flip_line2_walk_len = 0
        for point in line2:
            point_geom = Point(point)
            proj_point = find_nearest_projection_on_polyline(point_geom, line1_geom)
            if len(proj_point) != 0:
                flip_line2_walk_len += line1_geom.project(Point(proj_point[0]))

        if flip_line2_walk_len < line2_walk_len:
            return line2_flip
        else:
            return line2

    if flip_line2_traj_len < line2_traj_len:
        return line2_flip
    else:
        return line2


def get_line_lineList_max_intersection(merged_lines, line, thickness=4):
    pre_line = merged_lines[-1]     # 没有相交的情况返回这个
    max_iou = 0
    merged_line_index = 0
    for line_index, one_merged_line in enumerate(merged_lines):
        line1 = LineString(one_merged_line)
        line2 = LineString(line)
        thick_line1 = line1.buffer(thickness)
        thick_line2 = line2.buffer(thickness)
        intersection = thick_line1.intersection(thick_line2)
        if intersection.area / thick_line2.area > max_iou:
            max_iou = intersection.area / thick_line2.area
            pre_line = np.array(line1.coords)
            merged_line_index = line_index
    return intersection, pre_line, merged_line_index


def iou_merge_divider(merged_lines, vec, car_trajectory, thickness=1):
    # intersection : the intersection area between the new line and the line in the merged_lines; is a polygon
    # pre_line : the line in merged_lines that has max IOU with the new line   vec与merged中重合部分最大的线
    intersection, pre_line, merged_line_index = get_line_lineList_max_intersection(merged_lines, vec, thickness)
    # align new line with the line in the merged_lines so that points on two lines are traversed in the same direction
    vec = algin_l2_with_l1(pre_line, vec)

    line1 = LineString(pre_line)
    line2 = LineString(vec)

    # get the intersection points between IOU area and two lines
    # pdb.set_trace()
    line1_intersect_start, line1_intersect_end = get_intersection_point_on_line(pre_line, intersection)
    # start: [-23.60215759,   0.96279526];   end: [-22.20708466,   1.02379417]
    line2_intersect_start, line2_intersect_end = get_intersection_point_on_line(vec, intersection)
    # start: [-23.39431953,   1.12340546];   end: [-21.21332039,   1.12847144]
    # If no intersection points are found, use the last point of the line1 and the first point of the line2 as the
    # intersection points --> this is a corner case that we will connect the two lines head to tail directly
    if line1_intersect_start is None or line1_intersect_end is None or line2_intersect_start is None or line2_intersect_end is None:
        line1_intersect_start = Point(pre_line[-1])
        line1_intersect_end = Point(pre_line[-1])
        line2_intersect_start = Point(vec[0])
        line2_intersect_end = Point(vec[0])

    # merge the points on line2's intersection area towards line1
    merged_line2_points = merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end,
                                                car_trajectory)
    # merge the points on line1's intersection area towards line2
    merged_line1_points = merge_l2_points_to_l1(line2, line1, line1_intersect_start, line1_intersect_end,
                                                car_trajectory)

    # segment the lines based on the merged points (intersection area); split the line in to start segment and merged segment and end segment
    l2_start_segment, l2_end_segment = segment_line_based_on_merged_area(line2, merged_line2_points)
    l1_start_segment, l1_end_segment = segment_line_based_on_merged_area(line1, merged_line1_points)

    # choose the longer segment between line1 and line2 to be the final start segment and end segment
    start_segment = get_longer_segmenent_to_merged_points(l1_start_segment, l2_start_segment, merged_line2_points,
                                                          segment_type="start")
    end_segment = get_longer_segmenent_to_merged_points(l1_end_segment, l2_end_segment, merged_line2_points,
                                                        segment_type="end")
    merged_polyline = np.concatenate((start_segment, merged_line2_points, end_segment), axis=0)

    # update the merged_lines
    merged_lines[merged_line_index] = merged_polyline

    return merged_lines


def merge_divider(vecs=None, car_trajectory=None, thickness=1):
    merged_lines = []
    for vec in vecs:
        # if the merged_lines is empty, add the first line
        if len(merged_lines) == 0:
            merged_lines.append(vec)
            continue

        # thicken the vec (the new line) and the merged_lines calculate the max IOU between new line and merged_lines
        iou = []
        for one_merged_line in merged_lines:  # 唯一作用用于判断 max(iou) == 0
            line1 = LineString(one_merged_line)
            line2 = LineString(vec)
            thick_line1 = line1.buffer(thickness)
            thick_line2 = line2.buffer(thickness)
            intersection = thick_line1.intersection(thick_line2)
            iou.append(intersection.area / thick_line2.area)

        # If the max IOU is 0, add the new line to the merged_lines   同一条的不相交两段
        if max(iou) == 0:
            merged_lines.append(vec)
        # If IOU is not 0, merge the new line with the line in the merged_lines
        else:
            merged_lines = iou_merge_divider(merged_lines, vec, car_trajectory, thickness=thickness)

    return merged_lines


def plot_car(prev2curr_matrix, i, num_samples):
    car_img = Image.open('/home/sun/Bev/maptracker/resources/car-orange.png')
    rotation_degrees = np.degrees(np.arctan2(prev2curr_matrix[:3, :3][1, 0], prev2curr_matrix[:3, :3][0, 0]))
    car_center = get_prev2curr_vectors(np.array((0, 0)).reshape(1, 1, 2), prev2curr_matrix).squeeze()
    translation = transforms.Affine2D().translate(car_center[0], car_center[1])
    rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
    rotation_translation = rotation + translation
    faded_rate = np.linspace(0.2, 1, num=8)
    if i % 5 == 0 or i == num_samples - 1:
        plt.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation + plt.gca().transData,
                   alpha=faded_rate[int(i/5)])


def plot_gt_global_unmerged(selected_samples, anno_path, ax):
    curr_sample = selected_samples[-1]
    curr_anno_data = np.load(osp.join(anno_path, curr_sample), allow_pickle=True)
    curr_anno_data_dict = {key: curr_anno_data[key].tolist() for key in curr_anno_data.files}
    curr_input_dict = curr_anno_data_dict["input_dict"]
    curr_g2e_matrix = get_matrix(curr_input_dict, True)

    rand_points, prev2curr_gt, prev2curr_gt_types = [], [], []
    # print("start plot gt global as the background ")
    sample_load_start = time.time()
    for i, sample in enumerate(selected_samples[-2::-1]):
        sample_anno_data = np.load(osp.join(anno_path, sample), allow_pickle=True)
        sample_anno_data_dict = {key: sample_anno_data[key].tolist() for key in sample_anno_data.files}
        sample_input_dict = sample_anno_data_dict["input_dict"]
        ego_points = sample_anno_data_dict["ego_points"]

        ego_pts, ego_types = [], []
        for item in ego_points:
            pts_type = item['type']
            pts = item['pts'][:, :2]
            polyline = LineString(pts)
            distances = np.linspace(0, polyline.length, 20)
            sampled_points = np.array([list(polyline.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2)
            ego_pts.append(sampled_points)
            ego_types.append(pts_type)

        prev_e2g_matrix = get_matrix(sample_input_dict, False)
        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
        prev2curr_gt_vectors = get_prev2curr_vectors(ego_pts, prev2curr_matrix)
        prev2curr_gt.append(prev2curr_gt_vectors)
        prev2curr_gt_types.append(ego_types)

        plot_start = time.time()
        x_all, y_all, all_lines = [], [], []
        for m in range(len(prev2curr_gt_types)):
            prev2curr_gt_vector = prev2curr_gt[m]
            prev2curr_gt_type = prev2curr_gt_types[m]
            for n in range(prev2curr_gt_vector.shape[0]):
                pts = prev2curr_gt_vector[n]
                # pts_type = prev2curr_gt_type[n]
                # plot_line(pts, colors[np.int32(pts_type) + 1])
                # x_all.extend([pts[i][0], pts[i + 1][0]] for i in range(len(pts) - 1))
                # y_all.extend([pts[i][1], pts[i + 1][1]] for i in range(len(pts) - 1))
                segments = [[(pts[i][0], pts[i][1]), (pts[i + 1][0], pts[i + 1][1])] for i in range(len(pts) - 1)]
                all_lines.extend(segments)

        # plt.plot(x_all, y_all, c="#666666", linewidth=7)
        lc = LineCollection(all_lines, colors="#666666", linewidths=4)
        # fig, ax = plt.subplots()
        ax.add_collection(lc)
    #     plot_end = time.time()
    #     print(f"Plotting time: {plot_end - plot_start:.4f} seconds")
    #
    # # Measure total time
    # end_time = time.time()
    # print(f"Total execution time: {end_time - sample_load_start:.4f} seconds")


def get_map_size(selected_samples, anno_path, output_path, save_path, vis_pred_unmerged):
    colors = {1: 'red', 2: "blue", 3: 'green'}
    curr_sample = selected_samples[-1]
    curr_anno_data = np.load(osp.join(anno_path, curr_sample), allow_pickle=True)
    curr_anno_data_dict = {key: curr_anno_data[key].tolist() for key in curr_anno_data.files}
    curr_input_dict = curr_anno_data_dict["input_dict"]

    curr_data = np.load(osp.join(output_path, 'results', curr_sample), allow_pickle=True)
    curr_res = dict(curr_data['dt_res'].tolist())
    curr_map_points = ska_points(curr_res["map"])
    curr_label = curr_res["pred_label"]

    curr_g2e_matrix = get_matrix(curr_input_dict, True)

    fig = plt.figure(figsize=(200, 70))
    ax = fig.add_subplot(1, 1, 1)
    # plt.xlim(-30, 30)
    # plt.ylim(-15, 15)
    print("start plot Pred unmerged")
    rand_points, prev2curr_gt, prev2curr_gt_types = [], [], []
    for i, sample in enumerate(selected_samples[-2::-1]):
        sample_data = np.load(osp.join(output_path, 'results', sample), allow_pickle=True)
        sample_pred_res = dict(sample_data['dt_res'].tolist())
        sample_pred_map_points = ska_points(sample_pred_res["map"])
        sample_pred_label = sample_pred_res["pred_label"][1:]
        # GT
        sample_anno_data = np.load(osp.join(anno_path, sample), allow_pickle=True)
        sample_anno_data_dict = {key: sample_anno_data[key].tolist() for key in sample_anno_data.files}
        sample_input_dict = sample_anno_data_dict["input_dict"]
        ego_points = sample_anno_data_dict["ego_points"]
        ego_pts, ego_types = [], []
        for item in ego_points:
            pts_type = item['type']
            pts = item['pts'][:, :2]
            polyline = LineString(pts)
            distances = np.linspace(0, polyline.length, 50)
            sampled_points = np.array([list(polyline.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            ego_pts.append(sampled_points)
            ego_types.append(pts_type)

        prev_e2g_matrix = get_matrix(sample_input_dict, False)
        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
        # pdb.set_trace()
        prev2curr_pred_vectors = get_prev2curr_vectors(sample_pred_map_points, prev2curr_matrix, )
        prev2curr_gt_vectors = get_prev2curr_vectors(ego_pts, prev2curr_matrix)
        prev2curr_gt.append(prev2curr_gt_vectors)
        prev2curr_gt_types.append(ego_types)

        for j, points in enumerate(prev2curr_pred_vectors):
            rand_points.append(np.array(points[0]))
            rand_points.append(np.array(points[-1]))
            x = np.array([pt[0] for pt in points])
            y = np.array([pt[1] for pt in points])
            ax.plot(x, y, '-', color=colors[np.int32(sample_pred_label[j])], linewidth=8, markersize=10, alpha=1)
            # plot_line(points, colors[np.int32(sample_pred_label[j])], )

    # plot_car(prev2curr_matrix, i, len(selected_samples)-1)
    if vis_pred_unmerged:
        map_path = osp.join(save_path, 'Pred_unmerged.png')
        plt.gca().set_aspect('equal')
        # plt.axis('off')
        plt.savefig(map_path, bbox_inches='tight', format='png', dpi=40)
        print(f'Saved {map_path}')
        plt.close()

    print("start plot GT unmerged")
    fig = plt.figure(figsize=(200, 70))
    ax = fig.add_subplot(1, 1, 1)
    for m in range(len(prev2curr_gt_types)):
        prev2curr_gt_vector = prev2curr_gt[m]
        prev2curr_gt_type = prev2curr_gt_types[m]
        for n in range(prev2curr_gt_vector.shape[0]):
            pts = prev2curr_gt_vector[n]
            pts_type = prev2curr_gt_type[n]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, '-', color=colors[np.int32(pts_type)+1], linewidth=8, markersize=10, alpha=1)
            # plot_line(pts, colors[np.int32(pts_type)+1], linewidth=3)
    if "old" not in anno_path:
        map_path = osp.join(save_path, 'GT_unmerged.png')
    else:
        map_path = osp.join(save_path, 'GT_unmerged_old.png')
    plt.gca().set_aspect('equal')
    # plt.axis('off')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=40)
    plt.close()
    print(f'Saved {map_path}')

    for points in curr_map_points:
        rand_points.append(np.array(points[0]))
        rand_points.append(np.array(points[-1]))
    ss = np.stack(rand_points)
    s1 = np.array([np.floor(min(ss[:, 0])), np.ceil(max(ss[:, 0]))], dtype=int)
    s2 = np.array([np.floor(min(ss[:, 1])), np.ceil(max(ss[:, 1]))], dtype=int)
    s3 = np.array([(s1[1]-s1[0]), (s2[1]-s2[0])], dtype=float)
    s4 = np.array([s1[0], s2[0]], dtype=float)
    # pdb.set_trace()
    prev2curr_points_dict = points_list_to_dict(prev2curr_pred_vectors, sample_pred_label)    # 0.frame points
    return s3, s4, prev2curr_points_dict


def resize_image(image_path):
    with Image.open(image_path) as img:
        new_width = (img.width + 15) // 16 * 16
        new_height = (img.height + 15) // 16 * 16
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img.convert("RGBA")


def save_as_video(image_list, mp4_output_path, scale=None):
    mp4_output_path = mp4_output_path.replace('.gif', '.mp4')
    # images = [Image.fromarray(imageio.imread(img_path)).convert("RGBA") for img_path in image_list]
    images = [resize_image(img_path) for img_path in image_list]
    if scale is not None:
        w, h = images[0].size
        images = [img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS) for img in images]
    # images = [Image.new('RGBA', images[0].size, (255, 255, 255, 255))] + images

    try:
        imageio.mimsave(mp4_output_path, images, format='MP4', fps=1)
    except ValueError:  # in case the shapes are not the same, have to manually adjust
        resized_images = [img.resize(images[0].size, Image.Resampling.LANCZOS) for img in images]
        print('Size not all the same, manually adjust...')
        imageio.mimsave(mp4_output_path, resized_images, format='MP4', fps=10)
    print("mp4 saved to : ", mp4_output_path)


def vis_pred_local(selected_samples, save_path, output_path, anno_path, horizon=False):
    car_img = Image.open('/home/sun/Bev/maptracker/resources/car-orange.png')
    colors = {1: 'red', 3: "green", 2: 'blue'}

    vis_local_path = os.path.join(save_path, 'local_pred')
    if not os.path.exists(vis_local_path):
        os.makedirs(vis_local_path, exist_ok=True)

    for k, sample in enumerate(selected_samples):
        if horizon:
            plt.figure(figsize=(30, 15))
            plt.xlim(-31, 31)
            plt.ylim(-16, 16)
        else:
            plt.figure(figsize=(15, 30))
            plt.ylim(-31, 31)
            plt.xlim(-16, 16)
        curr_data = np.load(osp.join(output_path, 'results', sample), allow_pickle=True)
        curr_res = dict(curr_data['dt_res'].tolist())
        map_points = ska_points(curr_res["map"], horizon)
        label = curr_res["pred_label"][1:]
        # pdb.set_trace()
        sample_anno_data = np.load(osp.join(anno_path, sample), allow_pickle=True)   # gt
        sample_anno_data_dict = {key: sample_anno_data[key].tolist() for key in sample_anno_data.files}
        ego_points = sample_anno_data_dict["ego_points"]
        for item in ego_points:
            pts = item['pts']
            if horizon:
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
            else:
                x = np.array([-pt[1] for pt in pts])
                y = np.array([pt[0] for pt in pts])
            plt.plot(x, y, c="#666666", linewidth=2)

        for j, points in enumerate(map_points):
            x = np.array([pt[0] for pt in points])
            y = np.array([pt[1] for pt in points])
            if horizon:
                plt.plot(x, y, c=colors[int(label[j])], linewidth=4)
            else:
                plt.plot(y, x, c=colors[int(label[j])], linewidth=4)

        plt.imshow(car_img, extent=[-2.2, 2.2, -2, 2])
        map_path = osp.join(vis_local_path, f'{k}.png')
        plt.gca().set_aspect('equal')
        # plt.axis('off')
        plt.savefig(map_path, bbox_inches='tight', format='png', dpi=100)
        print(f'Saved {map_path}')
        plt.close()


def vis_gt_local(selected_samples, save_path, anno_path, crossing_anno_path, vis_bemapnet=True, vis_maptracker=False):
    car_img = Image.open('/home/sun/Bev/maptracker/resources/car-orange.png')
    colors = {1: 'red', 3: "green", 2: 'blue', "ped_crossing": "b", "divider": "r", "boundary": "g"}

    if "old" in anno_path:
        vis_local_path = os.path.join(save_path, 'local_gt_old')
    elif "80" in anno_path:
        vis_local_path = os.path.join(save_path, 'local_gt_80_x_50')
    else:
        vis_local_path = os.path.join(save_path, 'local_gt')
    if not os.path.exists(vis_local_path):
        os.makedirs(vis_local_path, exist_ok=True)

    vis_mt_local_path = os.path.join(save_path, 'maptracker_local_gt')
    # pdb.set_trace()
    if not os.path.exists(vis_mt_local_path) and vis_maptracker:
        os.makedirs(vis_mt_local_path, exist_ok=True)

    for k, sample in enumerate(selected_samples):
        plt.figure(figsize=(20, 10))
        if "80" not in anno_path:
            plt.xlim(-30, 30)
            plt.ylim(-15, 15)
        else:
            plt.xlim(-41, 41)
            plt.ylim(-26, 26)
        sample_anno_data = np.load(osp.join(anno_path, sample), allow_pickle=True)    # crossing 80 X 50
        sample_anno_data_dict = {key: sample_anno_data[key].tolist() for key in sample_anno_data.files}
        # crossing_sample_anno_data = np.load(osp.join(crossing_anno_path, sample), allow_pickle=True)
        # crossing_sample_anno_data_dict = {key: crossing_sample_anno_data[key].tolist() for key in crossing_sample_anno_data.files}
        ego_points = sample_anno_data_dict["ego_points"]
        # map_geoms = crossing_sample_anno_data_dict["input_dict"]["map_geoms"]

        if vis_bemapnet:
            for item in ego_points:
                pts = item['pts']
                plot_line(pts, colors[int(item['type'] + 1)])

            plt.imshow(car_img, extent=[-2.2, 2.2, -2, 2])
            map_path = osp.join(vis_local_path, f'{k}.png')
            plt.gca().set_aspect('equal')
            # plt.tight_layout()
            ax = plt.gca()
            # 设置边框为虚线
            for spine in ax.spines.values():
                spine.set_linestyle((0, (5, 10)))  # 设置虚线
                spine.set_linewidth(1.5)  # 可选：调整线宽
            plt.savefig(map_path, bbox_inches='tight', format='png', dpi=400)
            print(f'Saved {map_path}')
            plt.close()

        if vis_maptracker:
            plt.figure(figsize=(12, 6))
            plt.xlim(-31, 31)
            plt.ylim(-16, 16)
            for key, value in map_geoms.items():
                for pts in value:
                    for i in range(pts.shape[0] - 1):
                        plt.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], c=colors[key], linewidth=1)
                        if key == "ped_crossing":
                            plt.scatter(pts[i][0], pts[i][1], c=colors[key], s=4)
            plt.imshow(car_img, extent=[-2.2, 2.2, -2, 2])
            plt.tight_layout()
            map_path = osp.join(vis_mt_local_path, f'{k}.png')
            plt.savefig(map_path, bbox_inches='tight', format='png', dpi=400)
            plt.close()


def read_file_to_list(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [f"{line.strip()}.npz" for line in f]  # 返回列表


def main():
    # output_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_av2_res50/2024-09-12/evaluation"
    output_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_av2_res50_geosplit_interval_1/1004_perfect/evaluation_epoch_5"
    # anno_path = '/home/sun/Bev/BeMapNet/data/argoverse2/maptracker_interval_1'
    # anno_path = '/home/sun/Bev/maptracker/datasets/av2/80_X_50/val'
    anno_path = '/home/sun/Bev/BeMapNet/data/argoverse2/geo_interval_1'
    # anno_path = '/home/sun/Bev/BeMapNet/data/argoverse2/geosplits_interval_1_old'
    crossing_anno_path = '/home/sun/Bev/BeMapNet/data/argoverse2/geosplits_interval1_old'
    car_img = Image.open('/home/sun/Bev/maptracker/resources/car-orange.png')
    geo_val = read_file_to_list("/home/sun/Bev/BeMapNet/assets/splits/argoverse2/MapTacker_geosplit_val_scene_sample.txt")
    old_val = read_file_to_list("/home/sun/Bev/BeMapNet/assets/splits/argoverse2/MapTacker_oldsplit_val_scene_sample.txt")
    random_files = random.sample(geo_val, 20)
    pdb.set_trace()
    vis_path = '/home/sun/Bev/BeMapNet/visual_global/'
    scene = "0526e68e-2ff1-3e53-b0f8-45df02e45a93"
    scene = '6c932547-4c11-31d7-b8ef-0c16a13dbfc3'     # 02a00399-3857-444e-8db3-a8f58489c394
    colors = {'divider': 'red', 'boundary': "green", 'crossing': 'blue'}
    cfg = dict(canvas_size=(200, 100), thickness=3,)
    cutting = True
    Vis_pred_local = True
    Vis_gt_local = False
    vis_pred_unmerged = False

    if 'geo' in output_path:
        if cutting:
            save_path = os.path.join(vis_path, 'geosplit', 'av2', scene)
        else:
            save_path = os.path.join(vis_path, 'geosplit', 'av2', f"{scene}_without_cutting")
    else:
        if cutting:
            save_path = os.path.join(vis_path, 'original', 'av2', scene)
        else:
            save_path = os.path.join(vis_path, 'original', 'av2', f"{scene}_without_cutting")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    matching_files = []
    # for filename in os.listdir(osp.join(output_path, 'results')):
    #     if filename.startswith(scene):
    #         matching_files.append(filename)

    if "geo" in output_path:
        for filename in geo_val:
            if filename.startswith(scene):
                matching_files.append(filename)
    else:
        for filename in old_val:
            if filename.startswith(scene):
                matching_files.append(filename)
    # pdb.set_trace()

    selected_samples = sorted(matching_files)[::1]  # interval 4
    # pdb.set_trace()
    if Vis_pred_local:
        vis_pred_local(selected_samples, save_path, output_path, anno_path)
    if Vis_gt_local:
        vis_gt_local(selected_samples, save_path, anno_path, crossing_anno_path)
    pdb.set_trace()

    curr_sample = selected_samples[-1]
    curr_anno_data = np.load(osp.join(anno_path, curr_sample), allow_pickle=True)
    curr_anno_data_dict = {key: curr_anno_data[key].tolist() for key in curr_anno_data.files}
    curr_input_dict = curr_anno_data_dict["input_dict"]
    curr_g2e_matrix = get_matrix(curr_input_dict, True)

    # curr_data = np.load(osp.join(output_path, 'results', curr_sample), allow_pickle=True)
    # curr_res = dict(curr_data['dt_res'].tolist())
    # curr_map_points = ska_points(curr_res["map"])
    # curr_label = curr_res["pred_label"]

    roi_size, origin, merged_polyline = get_map_size(selected_samples, anno_path, output_path, save_path, vis_pred_unmerged)
    # pdb.set_trace()

    merged_per_frame_path = os.path.join(save_path, 'merged_per_frame')
    if not os.path.exists(merged_per_frame_path):
        os.makedirs(merged_per_frame_path, exist_ok=True)
    plt.figure(facecolor='lightgreen')
    fig = plt.figure(figsize=(roi_size[0] + 2, roi_size[1] + 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(origin[0], origin[0] + roi_size[0])
    ax.set_ylim(origin[1], origin[1] + roi_size[1])

    plot_gt_global_unmerged(selected_samples, anno_path, ax)
    start_time = time.time()
    for key, value in merged_polyline.items():
        if len(value) > 0:
            for polyline in value:
                # pdb.set_trace()
                # plot_line(polyline, colors[key], linewidth=8)
                x = np.array([pt[0] for pt in polyline])
                y = np.array([pt[1] for pt in polyline])
                ax.plot(x, y, '-', color=colors[key], linewidth=8, markersize=10, alpha=1)
    end_time = time.time()
    print(f"plot time: {end_time - start_time:.4f} seconds")
    map_path = osp.join(merged_per_frame_path, '0.png')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=60)
    print(f'Saved {map_path}')
    plt.close()

    # pdb.set_trace()
    car_trajectory = []
    for i, sample in enumerate(selected_samples[1:]):
        # plt.figure()
        start_time = time.time()
        plt.figure(facecolor='lightgreen')
        fig = plt.figure(figsize=(roi_size[0] + 2, roi_size[1] + 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(origin[0], origin[0] + roi_size[0])
        ax.set_ylim(origin[1], origin[1] + roi_size[1])

        sample_path = os.path.join(output_path, 'results', sample)

        # GT
        prev_anno_data = np.load(osp.join(anno_path, sample), allow_pickle=True)
        prev_anno_data_dict = {key: prev_anno_data[key].tolist() for key in prev_anno_data.files}
        # ['input_dict', 'instance_mask', 'instance_mask8', 'semantic_mask', 'ctr_points', 'ego_points', 'map_vectors']
        prev_input_dict = prev_anno_data_dict["input_dict"]
        prev_gt_points = prev_anno_data_dict["ego_points"]

        # Pred
        pred_data = np.load(sample_path, allow_pickle=True)
        pred_res = dict(pred_data['dt_res'].tolist())
        prev_pred_map_points = ska_points(pred_res["map"])
        prev_pred_label = pred_res["pred_label"][1:]       # [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]  1:divider  2:crossing
        prev_e2g_matrix = get_matrix(prev_input_dict, False)
        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix

        rotation_degrees = np.degrees(np.arctan2(prev2curr_matrix[:3, :3][1, 0], prev2curr_matrix[:3, :3][0, 0]))
        car_center = get_prev2curr_vectors(np.array((0, 0)).reshape(1, 1, 2), prev2curr_matrix).squeeze()
        if i % 4 == 0 or i == len(selected_samples) - 1:
            car_trajectory.append([car_center, rotation_degrees])

        if cutting:
            index_list = []  # [[0, 0], [4, 0], [5, 0], [14, 41], [16, 62]]
            if i >= 0:   # von 0 +1 frame zu schneiden
                # if prev_pred_map_points
                for m, ins_pts in enumerate(prev_pred_map_points):
                    if ins_pts[-1][0] < -10:   # coord from large to small
                        for n, pt in enumerate(ins_pts):
                            if pt[0] < -10:
                                index_list.append([m, n])
                                break
            if len(index_list) != 0:
                for index in index_list:
                    ins_index = index[0]
                    pt_index = index[1]
                    if pt_index == 1:
                        pt_index += 1
                    if pt_index != 0:
                        curr_vec_polyline = LineString(prev_pred_map_points[ins_index][:pt_index])
                        distances = np.linspace(0, curr_vec_polyline.length, 100)
                        sampled_points = np.array(
                            [list(curr_vec_polyline.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                        prev_pred_map_points[ins_index] = sampled_points
                    else:
                        prev_pred_map_points[ins_index] = prev_pred_map_points[ins_index][:pt_index]

                k = 0
                for index in index_list:
                    if index[1] == 0:
                        prev_pred_map_points.pop(index[0] - k)
                        prev_pred_label.pop(index[0] - k)
                        k += 1

        prev2curr_pred_vectors = get_prev2curr_vectors(prev_pred_map_points, prev2curr_matrix, )
        # time2 = time.time()
        # print(f"time2: {time2 - start_time:.4f} seconds")
        plot_gt_global_unmerged(selected_samples, anno_path, ax)

        faded_rate = np.linspace(0.2, 1, num=10)
        for n, car_loc in enumerate(car_trajectory):
            translation = transforms.Affine2D().translate(car_loc[0][0], car_loc[0][1])
            rotation = transforms.Affine2D().rotate_deg(car_loc[1])
            rotation_translation = rotation + translation
            plt.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation + plt.gca().transData,
                   alpha=faded_rate[n])
        prev2curr_points_dict = points_list_to_dict(prev2curr_pred_vectors, prev_pred_label)
        # time3 = time.time()
        # print(f"time3: {time3 - time2:.4f} seconds")
        # before_match_time = time.time()
        match_result = match_frames(merged_polyline, prev2curr_points_dict, roi_size, origin, cfg)
        # after_match_time = time.time()
        # print("Match time: ", after_match_time - before_match_time)
        # 'crossing': ([0, 2, 1, 3, 4, 5], [0, 2, 1, 3, 4, 5]), 'boundary': ([0, -1, 1, 2], [0, 2, 3]),
        # 'divider': ([1, 2, 5, 6, 7, 8], [-1, 0, 1, -1, -1, 2, 3, 4, 5])}
        # print(match_result)
        # pdb.set_trace()
        for key, value in match_result.items():
            if len(value[0]) == 0:
                if len(value[1]) == 0:
                    continue
                else:
                    for k, index in enumerate(value[1]):
                        if index == -1:
                            merged_polyline[key] = np.vstack(
                                [merged_polyline[key], np.expand_dims(prev2curr_points_dict[key][k], axis=0)])
            else:
                for k, index in enumerate(value[0]):
                    merged_line = merged_polyline[key][k]
                    if index != -1:
                        merge_line = prev2curr_points_dict[key][index]
                        if key == 'divider' or 'boundary':
                            polylines_vecs = merge_divider([merged_line, merge_line], car_center)
                            # pdb.set_trace()
                            line = LineString(polylines_vecs[0])
                            line_length = line.length
                            distances = np.linspace(0, line_length, 100)
                            points = [line.interpolate(distance) for distance in distances]
                            merged_polyline[key][k] = np.array([(point.x, point.y) for point in points])

                    # pdb.set_trace()
                if len(value[1]) != 0:
                    for k, index in enumerate(value[1]):
                        if index == -1:
                            merged_polyline[key] = np.vstack([merged_polyline[key], np.expand_dims(prev2curr_points_dict[key][k], axis=0)])

            for n in range(len(merged_polyline[key])):
                polyline = merged_polyline[key][n]
                # plot_line(polyline, colors[key])
                x = np.array([pt[0] for pt in polyline])
                y = np.array([pt[1] for pt in polyline])
                ax.plot(x, y, '-', color=colors[key], linewidth=8, markersize=10, alpha=1)

        map_path = osp.join(merged_per_frame_path, f'{i+1}.png')
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.savefig(map_path, bbox_inches='tight', format='png', dpi=60)
        end_time = time.time()
        print(f'Saved {map_path}, plot time: {end_time - start_time:.4f} seconds')
        plt.close()

    image_list = [merged_per_frame_path + f'/{frame_timestep}.png' for frame_timestep in range(len(selected_samples))]
    gif_output_path = merged_per_frame_path + '/vis.gif'
    save_as_video(image_list, gif_output_path)


if __name__ == "__main__":
    main()
