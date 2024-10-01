import pandas as pd

class DataLoader:
    def __init__(self, file_path, file_type="csv"):
        self.file_path = file_path
        self.file_type = file_type
    
    def load_data(self):
        if self.file_type == "csv":
            return pd.read_csv(self.file_path)
        elif self.file_type == "excel":
            return pd.read_excel(self.file_path)
        else:
            raise ValueError("不支持的文件类型")
