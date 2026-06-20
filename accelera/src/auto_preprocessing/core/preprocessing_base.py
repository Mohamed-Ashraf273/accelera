class PreprocessingBase:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        if folder_path is None:
            raise ValueError("folder_path cannot be None")

    def common_preprocessing(self):
        raise NotImplementedError("Must implement common_preprocessing method.")
