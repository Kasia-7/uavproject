# main.py
import numpy as np
from utils.utils import get_points,associate_point_with_trajectory,prune_trajectories
from utils.data_loader import DataLoader
from utils.preprocessing import MissingValueFillerPreprocessing


def main():
    # 1. 加载数据
    loader = DataLoader(file_path='data/点迹数据2-公开提供.xlsx', file_type='excel')
    data = loader.load_data()

    # 2. 数据预处理
    preprocessor = MissingValueFillerPreprocessing()
    preprocessed_data = preprocessor.preprocess(data)

    # 3. 航迹列表，储存航迹
    trajectorys = []

    # 4. 按圈数为批次对数据进行处理
    steps = sorted(preprocessed_data['圈数'].unique())
    for step in steps:
        # 先将所有航迹的未更新次数加1
        for track in trajectorys:
            track.update_unupdates()
        # 将所有的数据创建相应的点迹对象
        print(preprocessed_data)
        current_data = preprocessed_data[preprocessed_data['圈数'] == step]
        print(current_data)
        points = get_points(current_data)
        # 对每个点进行处理
        associate_point_with_trajectory(points,trajectorys)
        # 判断各个航迹的未更新次数是否达到阈值
        prune_trajectories(trajectorys)
        # 通过所有航迹的当前点计算当前质心

        # 通过所有航迹的预测点计算预测质心

        # 通过航迹数量返回当前无人机规模（可以简单粗暴地先取所有的平均值看看效果，要获得较为准确的规模还需要多加一步判断当前航迹数量是否稳定）

if __name__ == '__main__':
    main()
