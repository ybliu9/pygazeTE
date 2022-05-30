import numpy as np
import pandas as pd
import cv2
import os
import time
import pytesseract as pt
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import contingency_matrix

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

from pygazete.table_structure import *
from pygazete.ocr import *
from pygazete.preprocess import pred_coords

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    

        
        