import pandas as pd

from accelera.src.custom.transformer import CustomTransformer


class Flatten1DTransform(CustomTransformer):
    def __init__(self,func):
        self.func = func
    
    def fit(self, X, y=None):
       return self
   
    def transform(self, X):
      return self.func(X)
   
    def get_feature_names_out(self, input_features=None):
        return input_features
