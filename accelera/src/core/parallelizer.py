import pickle

from accelera.src.utils.code_utils import extract_features
from accelera.src.utils.code_utils import feature_dict_to_vector
from accelera.src.utils.code_utils import generate_omp_pragma


class Parallelizer:
    def __init__(self):
        self.encoder = self._load_encoder()
        self.classifier = self._load_classifier()

    def _load_encoder(self):
        with open(
            "/home/mohamed-ashraf/Desktop/projects/accelera/models/label_encoder.pkl",
            "rb",
        ) as f:
            return pickle.load(f)

    def _load_classifier(self):
        with open(
            "/home/mohamed-ashraf/Desktop/projects/accelera/models/model.pkl",
            "rb",
        ) as f:
            return pickle.load(f)

    def parallelize(self, code: str) -> str:
        features = extract_features(code)
        embedding = feature_dict_to_vector(features)
        embedding = embedding.reshape(1, -1)
        pred = self.model.predict(embedding)
        pred_class = self.encoder.inverse_transform(pred)
        predicted_pragma = generate_omp_pragma(code, pred_class)
        return predicted_pragma


parallelizer = Parallelizer()
