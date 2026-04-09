import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from accelera.src.automl.core.training_tabular_preprocessing_base import (
    TrainingTabularPreprocessingBase,
)
from accelera.src.automl.wrappers.tabular_preprocessing_report import (
    TabularPreprocessingReport,
)
from accelera.src.automl.wrappers.target_classification import TargetClassification
from accelera.src.automl.wrappers.text_graph import TextGraph
from accelera.src.utils.preprocessing import save_pickle


class TextTrainingPreprocessing(TrainingTabularPreprocessingBase):
    def __init__(
        self,
        df,
        target_col: str,
        text_col: str,
        folder_path=None,
        val_size=0.2,
        random_state=42,
        tfidf_max_features=5000,
        tfidf_ngram=(1, 2),
        tfidf_max_df=0.85,
        tfidf_min_df=3,
    ):
        super().__init__(df, target_col, val_size, random_state, folder_path)
        self.text_col = text_col
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram = tfidf_ngram
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_min_df = tfidf_min_df
        if target_col == text_col:
            raise ValueError("target column and text column must not be the same")
        if text_col not in df.columns:
            raise ValueError(
                f"Text column {text_col} not found in dataframe columns"
            )
        if df[text_col].dtype != "object":
            raise ValueError(f"Text column {text_col} must be of type object")
        if not (
            np.issubdtype(self.target_type, np.integer)
            or self.target_type == "object"
            or self.target_type == "bool"
        ):
            raise ValueError(
                "Target must be integer or object fro classification problem"
            )
        self.info = {"target_col": target_col, "text_col": text_col}
        unwanted_columns = []
        for col in self.df.columns:
            if col not in [self.target_col, self.text_col]:
                unwanted_columns.append(col)
        self.df.drop(columns=unwanted_columns, inplace=True)

        nltk.download("punkt")
        nltk.download("stopwords")

    def make_graphs(self, X_train, y_train):
        new_df = X_train.copy()
        new_df[self.target_col] = y_train
        self.report_data["graphs"] = {
            "folder_path": self.folder_path,
            "images_name": [],
        }
        text_graph = TextGraph(
            new_df,
            col_name=self.text_col,
            target_name=self.target_col,
            folder_path=self.folder_path,
        )
        text_graph.build_graph()
        self.report_data["graphs"]["images_name"].append(self.text_col)

        target_graph = TargetClassification(
            new_df,
            col_name=self.target_col,
            target_name=self.target_col,
            folder_path=self.folder_path,
        )
        target_graph.build_graph()
        self.report_data["graphs"]["images_name"].append(self.target_col)

    def custom_text_tokenizer(self, text):
        stop_words = set(stopwords.words("english")) - {"not", "no"}
        stem = PorterStemmer()
        text = text.lower().strip()
        text = re.sub(r"’", "'", text)
        text = re.sub(r"can[o']t", "can not", text)
        text = re.sub(r"won[o']t", "will not", text)
        text = re.sub(r"shan[o']t", "shall not", text)
        text = re.sub(r"n't\b", " not ", text)
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
        text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
        text = re.sub(r"[^a-zA-Z0-9']", " ", text)
        text = re.sub(r"\s+", " ", text)
        tokens = [word.strip("'").strip('"').strip() for word in text.split()]
        tokens = [
            stem.stem(word)
            for word in tokens
            if word not in stop_words and len(word) > 1
        ]
        return tokens

    def features_preprocessing(self, X_train, X_val):
        X_train[self.text_col] = X_train[self.text_col].fillna("")
        X_val[self.text_col] = X_val[self.text_col].fillna("")
        self.report_data["preprocessing"].append(
            {
                "col_name": self.text_col,
                "col_type": "Text",
                "col_preprocessing": [
                    "Fill nulls with empty text",
                    "Cleaning & Tokeniztion & Stemming",
                    "Tfidf Vectorizer",
                ],
            }
        )
        X_train_head = (
            X_train[self.text_col]
            .head()
            .reset_index(drop=True)
            .to_frame(name=self.text_col)
        )
        X_val_head = (
            X_val[self.text_col]
            .head()
            .reset_index(drop=True)
            .to_frame(name=self.text_col)
        )
        X_train_head_cleaned = (
            X_train_head[self.text_col]
            .apply(lambda x: " ".join(self.custom_text_tokenizer(x)))
            .to_frame(
                name=f"{self.text_col} After Cleaning & Tokeniztion & Stemming"
            )
        )
        X_val_head_cleaned = (
            X_val_head[self.text_col]
            .apply(lambda x: " ".join(self.custom_text_tokenizer(x)))
            .to_frame(
                name=f"{self.text_col} After Cleaning & Tokeniztion & Stemming"
            )
        )
        self.report_data["after_preprocessing"] = {
            "X_train_processed": pd.concat(
                [X_train_head, X_train_head_cleaned], axis=1
            ),
            "X_val_processed": pd.concat([X_val_head, X_val_head_cleaned], axis=1),
        }
        training_preprocessor = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram,
            max_df=self.tfidf_max_df,
            min_df=self.tfidf_min_df,
            tokenizer=self.custom_text_tokenizer,
            token_pattern=None,
            lowercase=False,
        )

        X_train_preprocessed = training_preprocessor.fit_transform(
            X_train[self.text_col]
        )
        X_val_preprocessed = training_preprocessor.transform(X_val[self.text_col])

        save_pickle(
            self.folder_path, training_preprocessor, "training_preprocessor.pkl"
        )
        return X_train_preprocessed, X_val_preprocessed

    def target_preprocessing(self, y_train, y_val):
        self.report_data["preprocessing"].append(
            {
                "col_name": self.target_col,
                "col_type": "Classification",
                "col_preprocessing": [
                    "Fill missing with most frequent",
                    "Label encoding",
                ],
            }
        )
        mode = y_train.mode()[0]
        self.info["target_mode"] = mode
        y_train = y_train.fillna(mode)
        y_val = y_val.fillna(mode)
        label_encoder = LabelEncoder()
        y_train_preprocessed = label_encoder.fit_transform(y_train)
        y_val_preprocessed = label_encoder.transform(y_val)
        self.report_data["after_preprocessing"]["y_train_processed"] = pd.DataFrame(
            y_train_preprocessed, columns=[self.target_col]
        ).head()
        self.report_data["after_preprocessing"]["y_val_processed"] = pd.DataFrame(
            y_val_preprocessed, columns=[self.target_col]
        ).head()
        save_pickle(self.folder_path, label_encoder, "target_preprocessor.pkl")
        save_pickle(self.folder_path, self.info, "data_info.pkl")
        return y_train_preprocessed, y_val_preprocessed

    def common_preprocessing(self):
        self.report_data["drop_columns"] = {"col_drop": None}
        self.data_overview()
        self.drop_duplicates()
        X_train, X_val, y_train, y_val = self.split_data()
        self.make_graphs(X_train, y_train)
        self.report_data["preprocessing"] = []
        X_train, X_val = self.features_preprocessing(X_train, X_val)
        y_train, y_val = self.target_preprocessing(y_train, y_val)
        report = TabularPreprocessingReport(
            self.folder_path, self.report_data, text_based=True
        )
        report.execute()
        return X_train, y_train, X_val, y_val
