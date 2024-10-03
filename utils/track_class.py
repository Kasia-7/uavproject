import math
import numpy as np
# 点迹类
class TrackPoint:
    def __init__(self, time, slant_range, azimuth_angle, elevation_angle, radial_velocity, cycle):
        """
        初始化点迹类，包含时间、斜距、方位角、俯仰角、径向速度和圈数。
        
        :param time: 时间
        :param slant_range: 斜距
        :param azimuth_angle: 方位角
        :param elevation_angle: 俯仰角
        :param radial_velocity: 径向速度
        :param cycle: 圈数
        """
        self.time = time
        self.slant_range = slant_range
        self.azimuth_angle = azimuth_angle
        self.elevation_angle = elevation_angle
        self.radial_velocity = radial_velocity
        self.cycle = cycle
        # 状态方程数组包含位置和速度信息
        self.state_equation = [self.slant_range, self.azimuth_angle, self.elevation_angle, self.radial_velocity]

    # 返回点迹对象的字符串表示
    def __str__(self):
        """
        返回点迹对象的字符串表示。
        """
        return (f"Time: {self.time}, "
                f"Slant Range: {self.slant_range}, "
                f"Azimuth Angle: {self.azimuth_angle}, "
                f"Elevation Angle: {self.elevation_angle}, "
                f"Radial Velocity: {self.radial_velocity}, "
                f"Cycle: {self.cycle}, "
                f"State Equation: {self.state_equation}")
    
    # 计算当前点迹与另一个点迹之间的距离
    def calculate_distance(self, other):
        """
        计算当前点迹与另一个点迹之间的距离。
        
        :param other: 另一个点迹对象
        :return: 两点之间的距离
        """
        if not isinstance(other, TrackPoint):
            raise ValueError("The argument must be an instance of TrackPoint")

        # 将角度从度转换为弧度
        azimuth1 = math.radians(self.azimuth_angle)
        elevation1 = math.radians(self.elevation_angle)
        azimuth2 = math.radians(other.azimuth_angle)
        elevation2 = math.radians(other.elevation_angle)

        # 计算两个点的笛卡尔坐标
        x1 = self.slant_range * math.cos(elevation1) * math.cos(azimuth1)
        y1 = self.slant_range * math.cos(elevation1) * math.sin(azimuth1)
        z1 = self.slant_range * math.sin(elevation1)

        x2 = other.slant_range * math.cos(elevation2) * math.cos(azimuth2)
        y2 = other.slant_range * math.cos(elevation2) * math.sin(azimuth2)
        z2 = other.slant_range * math.sin(elevation2)

        # 使用三维空间中两点之间的距离公式计算距离
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance
    
    #  根据方位角、俯仰角和径向速度计算xyz方向上的速度分量
    def calculate_velocity_components(self):
        """
        根据方位角、俯仰角和径向速度计算xyz方向上的速度分量。

        :param azimuth_angle: 方位角，从北方向顺时针测量的角度（度）
        :param elevation_angle: 俯仰角，物体相对于地平线的角度（度）
        :param radial_velocity: 径向速度，沿观察者到物体连线的速度分量
        :return: xyz方向上的速度分量
        """
        # 将角度从度转换为弧度
        azimuth_angle_rad = math.radians(self.azimuth_angle)
        elevation_angle_rad = math.radians(self.elevation_angle)

        # 计算vx, vy, vz速度分量
        vx = self.radial_velocity * math.cos(elevation_angle_rad) * math.cos(azimuth_angle_rad)
        vy = self.radial_velocity * math.cos(elevation_angle_rad) * math.sin(azimuth_angle_rad)
        vz = self.radial_velocity * math.sin(elevation_angle_rad)

        return vx, vy, vz
    
    # 将极坐标转换为直角坐标系中的xyz坐标
    def polar_to_cartesian(self):
        """
        将极坐标转换为直角坐标系中的xyz坐标。

        :param slant_range: 斜距，物体到观察点的直线距离
        :param azimuth_angle: 方位角，从北方向顺时针测量的角度（度）
        :param elevation_angle: 俯仰角，物体相对于地平线的角度（度）
        :return: 直角坐标系中的xyz坐标
            """
        # 将角度从度转换为弧度
        azimuth_angle_rad = math.radians(self.azimuth_angle)
        elevation_angle_rad = math.radians(self.elevation_angle)

        # 计算x, y, z坐标
        x = self.slant_range * math.cos(elevation_angle_rad) * math.cos(azimuth_angle_rad)
        y = self.slant_range * math.cos(elevation_angle_rad) * math.sin(azimuth_angle_rad)
        z = self.slant_range * math.sin(elevation_angle_rad)

        return x, y, z
    
# 航迹类
class Track:
    def __init__(self, point, max_unupdates=4, dt=3):
        """
        初始化航迹类，包含点迹列表、预测点迹和未更新次数。
        
        :param max_unupdates: 允许的最大未更新次数
        """
        self.track_points = []  # 存储点迹的列表
        self.predicted_track_point = None  # 预测的下一个点迹
        self.unupdates = 0  # 未更新次数
        self.max_unupdates = max_unupdates  # 允许的最大未更新次数
        self.dt = dt
        # 卡尔曼滤波器初始化
        x,y,z = point.polar_to_cartesian()
        vx,vy,vz = point.calculate_velocity_components()
        self.state_estimate = [x,y,z,vx,vy,vz]  # 状态估计[x, y, z, vx, vy, vz]
        self.covariance_estimate = np.eye(6)  # 状态估计协方差矩阵
        self.process_noise_covariance = np.diag([1, 1, 1, 0.1, 0.1, 0.1])  # 过程噪声协方差矩阵
        self.measurement_noise_covariance = np.diag([1, 1, 1, 0.1, 0.1, 0.1])  # 测量噪声协方差矩阵
        self.measurement_matrix = np.eye(6)  # 测量矩阵
        self.state_transition_matrix = np.eye(6)  # 状态转移矩阵
        self.state_transition_matrix[:3, 3:] = dt * np.eye(3)
        # 生成预测点
        self.track_points.append(point)
        self.predict_next_point()
        # 更新卡尔曼滤波器
        self.update_kalman_filter(point)
        # self.track_points.append(point)
        # self.predict_next_point()

    # 向航迹中添加一个新的点迹
    def add_track_point(self, track_point):
        """
        向航迹中添加一个新的点迹。
        
        :param track_point: 新的点迹对象
        """
        self.track_points.append(track_point)
        # 获得预测点
        self.predict_next_point()
        # 更新卡尔曼滤波器
        self.update_kalman_filter(track_point)
        # print('当前点数据：',track_point)
        # print('预测点数据：',self.predicted_track_point)
        
    # 更新未更新次数
    def update_unupdates(self):
        """
        更新未更新次数。
        """
        self.unupdates += 1

    # 检查航迹是否有效
    def is_track_valid(self):
        """
        检查航迹是否有效。
        
        :return: 如果未更新次数小于最大允许值，则返回True，否则返回False
        """
        return self.unupdates < self.max_unupdates
    
    # 判断航迹是否为稳定航迹
    def is_stable(self):
        """
        判断航迹是否为稳定航迹。

        :return: 如果航迹点数超过20，则为稳定航迹，返回True，否则返回False
        """
        return len(self.track_points) > 20
    
    # 通过卡尔曼滤波器预测航迹的下一个点迹
    def predict_next_point(self):
            """
            预测航迹的下一个点迹。
            """
            # 卡尔曼滤波器预测步骤
            self.state_estimate = np.dot(self.state_transition_matrix, self.state_estimate)
            state_estimate = get_full_polar_coordinates(self.state_estimate)
            self.covariance_estimate = np.dot(np.dot(self.state_transition_matrix, self.covariance_estimate), self.state_transition_matrix.T) + self.process_noise_covariance
            if len(self.track_points) > 0:
                last_point = self.track_points[-1]
                self.predicted_track_point = TrackPoint(
                    time=last_point.time + 3,
                    slant_range=state_estimate[0],
                    azimuth_angle=state_estimate[1],
                    elevation_angle=state_estimate[2],
                    radial_velocity=last_point.radial_velocity,
                    cycle=last_point.cycle + 1
                )
            else:
                self.predicted_track_point = None

    # 更新卡尔曼滤波器
    def update_kalman_filter(self, track_point):
        """
        更新卡尔曼滤波器。
        """
        # 将track_point转换为适合卡尔曼滤波的测量向量
        x,y,z = track_point.polar_to_cartesian()
        vx,vy,vz = track_point.calculate_velocity_components()
        measurement = [x,y,z,vx,vy,vz]
        # 卡尔曼滤波器更新步骤
        innovation = measurement - np.dot(self.measurement_matrix, self.state_estimate)
        innovation_cov = np.dot(np.dot(self.measurement_matrix, self.covariance_estimate), self.measurement_matrix.T) + self.measurement_noise_covariance
        kalman_gain = np.dot(np.dot(self.covariance_estimate, self.measurement_matrix.T), np.linalg.inv(innovation_cov))
        self.state_estimate = self.state_estimate + np.dot(kalman_gain, innovation)
        self.covariance_estimate = np.dot(np.eye(6) - np.dot(kalman_gain, self.measurement_matrix), self.covariance_estimate)


    # 返回航迹对象的字符串表示
    def __str__(self):
        """
        返回航迹对象的字符串表示。
        """
        track_points_str = "\n".join(str(point) for point in self.track_points)
        return (f"Track Points:\n{track_points_str}\n"
                f"Predicted Track Point: {self.predicted_track_point}\n"
                f"Unupdates: {self.unupdates}\n"
                f"Max Unupdates: {self.max_unupdates}")
    
def cartesian_to_polar(x,y,z):
    # 计算斜距
    slant_range = math.sqrt(x**2 + y**2 + z**2)

    # 计算方位角
    azimuth_angle = math.atan2(y, x)
    azimuth_angle_deg = math.degrees(azimuth_angle)

    # 计算俯仰角
    elevation_angle = math.asin(z / slant_range)
    elevation_angle_deg = math.degrees(elevation_angle)

    return slant_range, azimuth_angle_deg, elevation_angle_deg

def calculate_radial_velocity(x,y,z,vx,vy,vz,slant_range):
    # 计算径向速度
    # 径向速度是速度向量在从观察点到目标点连线上的投影
    radial_velocity = -((x * vx + y * vy + z * vz) / slant_range)
    return radial_velocity

def get_full_polar_coordinates(state):
    x,y,z,vx,vy,vz = state
    slant_range, azimuth_angle_deg, elevation_angle_deg = cartesian_to_polar(x,y,z)
    radial_velocity = calculate_radial_velocity(x,y,z,vx,vy,vz,slant_range)
    
    return slant_range, azimuth_angle_deg, elevation_angle_deg, radial_velocity