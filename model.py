import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


def ordinal_encoder(input_val, feats):
#   feat_val = list(1+np(len(feats)))
    feat_val = list(range(1, len(feats) + 1))
    feat_key=feats
    feat_dict = dict(zip(feat_key,feat_val))
    value = feat_dict[input_val]
    return value


def get_prediction(data):
    model_path = "model.sav"
    model = joblib.load(model_path)
    return model.predict(data)
    