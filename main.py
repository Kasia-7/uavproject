# main.py
import time
import numpy as np
from utils.utils import get_points, associate_point_with_trajectory, prune_trajectories, calculate_centroid,pred_calculate_centroid
from utils.data_loader import DataLoader
from utils.preprocessing import MissingValueFillerPreprocessing


def main():
    # 1. 加载数据
    loader = DataLoader(file_path='data/点迹数据3-公开提供.xlsx', file_type='excel')
    data = loader.load_data()

    # 2. 数据预处理
    preprocessor = MissingValueFillerPreprocessing()
    preprocessed_data = preprocessor.preprocess(data)

    # 3. 航迹列表，储存航迹
    trajectorys = []

    # 4. 按圈数为批次对数据进行处理
    steps = sorted(preprocessed_data['圈数'].unique())
    for step in steps:
        print(f'-----------------------------------------------------------------------{step}--------------------------------------------------------------------------------------------------------')
        # 先将所有航迹的未更新次数加1
        for track in trajectorys:
            track.update_unupdates()
        # 将所有的数据创建相应的点迹对象
        # print(preprocessed_data)
        current_data = preprocessed_data[preprocessed_data['圈数'] == step].values
        # print(current_data)
        points = get_points(current_data)
        # 对每个点进行处理
        associate_point_with_trajectory(points,trajectorys)
        # 将未匹配到的航迹先用预测点代替真实点
        for track in trajectorys:
            if track.unupdates != 0:
                track.add_track_point(track.predicted_track_point)
        # 判断各个航迹的未更新次数是否达到阈值，并返回当前群规模
        count = prune_trajectories(trajectorys)
        print("当前航迹总数",len(trajectorys))
        print("当前无人机规模为",count)
        # 通过所有航迹的当前点计算当前质心
        print("当前群中心为",calculate_centroid(trajectorys))
        # 通过所有航迹的预测点计算预测质心
        print("预测的群中心为",pred_calculate_centroid(trajectorys))
        # print("开始执行")
        # time.sleep(20)  # 暂停2秒
        # print("5秒后继续执行")
if __name__ == '__main__':
    main()
