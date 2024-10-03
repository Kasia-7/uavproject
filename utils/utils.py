import sys

import numpy as np
from pandas import Flags
import math

from utils.track_class import TrackPoint,Track

def get_points(current_data):
    """
    将给定的数据转换为 TrackPoint 对象的列表。
    
    :param current_data: 包含航迹点数据的列表，每个元素是一个包含时间、斜距、方位角、俯仰角、径向速度和圈数的元组或列表
    :return: TrackPoint 对象的列表
    """
    track_points = []
    for data in current_data:
        if len(data) != 6:
            raise ValueError("Each data point must contain exactly 6 elements")
        time, slant_range, azimuth_angle, elevation_angle, radial_velocity, cycle = data
        track_point = TrackPoint(time, slant_range, azimuth_angle, elevation_angle, radial_velocity, cycle)
        track_points.append(track_point)
    return track_points

def associate_point_with_trajectory(points, trajectories, distance_threshold=150.0):
    """
    将点与现有的航迹进行关联，如果无法关联，则创建新的航迹。

    :param point: 当前处理的点
    :param trajectories: 现有的航迹列表
    :param distance_threshold: 点与航迹关联的距离阈值
    :return: 更新后的航迹列表
    """
    # 存储未匹配的点
    unmatched_points = []
    # 关联矩阵，表示航迹是否被点匹配，防止航迹多选点
    track_flag = np.zeros(len(trajectories))
    # 如果航迹为空，所有点都设为未匹配
    count = 0
    if not trajectories:
        unmatched_points = points
    else:
        for j,point in enumerate(points):
            # 初始化数据
            track = None
            min_distance = sys.maxsize
            for i,trajectory in enumerate(trajectories):
                if track_flag[i]:
                    continue
                # 判断该航迹是否是离该点最短的航迹
                if point.calculate_distance(trajectory.predicted_track_point) < min_distance:
                    flag_i = i
                    track = trajectory
                    min_distance = point.calculate_distance(trajectory.predicted_track_point)
            # print(min_distance)
            # 如果预测的点与当前点的距离小于阈值，则将当前点添加到航迹中
            if track and point.calculate_distance(track.predicted_track_point) < distance_threshold:
                count = count + 1
                track_flag[flag_i] = 1
                track.add_track_point(point)
                # 当添加新点迹时，重置未更新次数
                track.unupdates = 0
            # 如果没有找到合适的航迹，先将点收集起来，最后统一创建新的航迹
            else:
                unmatched_points.append(point)
    # 对未匹配的点统一创建新的航迹            
    for point in unmatched_points:
        new_trajectory = Track(point)
        trajectories.append(new_trajectory)

def prune_trajectories(trajectories):
    """
    删除未更新次数达到阈值的航迹。

    :param trajectories: 航迹列表
    """
    trajectories[:] = [trajectory for trajectory in trajectories if trajectory.is_track_valid()]
    count = 0
    for track in trajectories:
        if track.is_stable():
            count = count + 1
    return count

# 计算所有航迹的当前点迹质心
def calculate_centroid(trajectorys):
    if not trajectorys:
        return [0, 0, 0]

    centroids = []
    for track in trajectorys:
        if len(track.track_points)>20 and track.track_points and track.track_points[-1]:
            azimuth_angle = track.track_points[-1].azimuth_angle
            elev_angle = track.track_points[-1].elevation_angle
            distance = track.track_points[-1].slant_range

            azimuth_rad = np.radians(azimuth_angle)
            elev_rad = np.radians(elev_angle)

            x = distance * np.cos(elev_rad) * np.cos(azimuth_rad)
            y = distance * np.cos(elev_rad) * np.sin(azimuth_rad)
            z = distance * np.sin(elev_rad)
            centroids.append([x, y, z])

    if not centroids:
        return [0, 0, 0]

    # 使用 NumPy 的 mean 函数计算所有质心坐标的平均值
    centroid_avg = np.mean(np.array(centroids), axis=0)
    return list(centroid_avg)

# 计算所有航迹的预测点迹质心
def pred_calculate_centroid(trajectorys):
    if not trajectorys:
        return [0, 0, 0]

    centroids = []
    for track in trajectorys:
        if len(track.track_points)>20 and track.predicted_track_point:
            azimuth_angle = track.predicted_track_point.azimuth_angle
            elev_angle = track.predicted_track_point.elevation_angle
            distance = track.predicted_track_point.slant_range

            azimuth_rad = np.radians(azimuth_angle)
            elev_rad = np.radians(elev_angle)

            x = distance * np.cos(elev_rad) * np.cos(azimuth_rad)
            y = distance * np.cos(elev_rad) * np.sin(azimuth_rad)
            z = distance * np.sin(elev_rad)
            centroids.append([x, y, z])

    if not centroids:
        return [0, 0, 0]

    # 使用 NumPy 的 mean 函数计算所有质心坐标的平均值
    centroid_avg = np.mean(np.array(centroids), axis=0)
    return list(centroid_avg)