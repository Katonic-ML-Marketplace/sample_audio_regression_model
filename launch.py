
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler

import pickle
import numpy as np 
from typing import Any, Union,Dict
import os
import json
import pickle
import tensorflow as tf
from tensorflow import keras

def loadmodel(logger):
    """Get model from cloud object storage."""
    logger.info("loading model")
    model = ""
    with open('audio_regression.pickle','rb') as f:
        model = pickle.load(f)
    logger.info("returning model object") 
    return model  

def preprocessing(df:np.ndarray,logger):
    """ Applies preprocessing techniques to the raw data"""
    ## in template keep this False by default, if its there then the return result will be other than False
    logger.info("no preprocessing")
    return False
    
def predict(features: np.ndarray,model:Any,logger) -> Dict[str, str]:
    """Predicts the results for the given inputs"""
    try:
        logger.info("model prediction")
        prediction=model.predict(features)
        logger.info("prediction successful")
        return prediction.tolist()
    except Exception as e:
        logger.info(e)
        return(e)
    
   
