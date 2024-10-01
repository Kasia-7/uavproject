from utils.track_class import TrackPoint,Track

def get_points(current_data):
    """
    将给定的数据转换为 TrackPoint 对象的列表。
    
    :param current_data: 包含航迹点数据的列表，每个元素是一个包含时间、斜距、方位角、俯仰角、径向速度和圈数的元组或列表
    :return: TrackPoint 对象的列表
    """
    track_points = []
    for data in current_data:
        print(data)
        if len(data) != 6:
            raise ValueError("Each data point must contain exactly 6 elements")
        time, slant_range, azimuth_angle, elevation_angle, radial_velocity, cycle = data
        track_point = TrackPoint(time, slant_range, azimuth_angle, elevation_angle, radial_velocity, cycle)
        track_points.append(track_point)
    return track_points

def associate_point_with_trajectory(points, trajectories, distance_threshold=10.0):
    """
    将点与现有的航迹进行关联，如果无法关联，则创建新的航迹。

    :param point: 当前处理的点
    :param trajectories: 现有的航迹列表
    :param distance_threshold: 点与航迹关联的距离阈值
    :return: 更新后的航迹列表
    """
    # 存储未匹配的点
    unmatched_points = []
    # 如果航迹为空，所有点都设为未匹配
    if trajectories:
        unmatched_points = points
    else:
        for point in points:
            for trajectory in trajectories:
                # 预测航迹的下一个点
                trajectory.predict_next_point()
                # 如果预测的点与当前点的距离小于阈值，则将当前点添加到航迹中
                if trajectory.predicted_track_point and point.calculate_distance(trajectory.predicted_track_point) < distance_threshold:
                    trajectory.add_track_point(point)
                # 如果没有找到合适的航迹，先将点收集起来，最后统一创建新的航迹
                else:
                    unmatched_points.append(point)

    # 对未匹配的点统一创建新的航迹            
    for point in unmatched_points:
        new_trajectory = Track()
        new_trajectory.add_track_point(point)
        trajectories.append(new_trajectory)

    return trajectories

def prune_trajectories(trajectories):
    """
    删除未更新次数达到阈值的航迹。

    :param trajectories: 航迹列表
    """
    # 创建一个迭代器，以便在遍历时删除航迹
    iterator = iter(trajectories)
    while True:
        try:
            trajectory = next(iterator)
            if trajectory.is_track_valid():
                # 如果未更新次数达到阈值，则从列表中删除
                trajectories.remove(trajectory)
            else:
                # 否则，继续检查下一个航迹
                continue
        except StopIteration:
            # 迭代完成，退出循环
            break
