from sklearn.preprocessing import StandardScaler

class PreprocessingStrategy:
    def preprocess(self, data):
        pass

class StandardScalerPreprocessing(PreprocessingStrategy):
    def preprocess(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

class MissingValueFillerPreprocessing(PreprocessingStrategy):
    def preprocess(self, data):
        return data.fillna(0)
