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
    
def pad_length(output, ref):
    output, ref = list(output), list(ref)
    l1, l2 = len(output), len(ref)
    if l1 == l2: return output
    elif l1 > l2:
        return output[:l2] if f1_score(ref, output[:l2], average='macro') >= f1_score(ref, output[l1-l2:], average='macro') else output[l1-l2:]
    else:  #pad length to output
        out1 = [' '] * (l2-l1) + output
        out2 = output + [' '] * (l2-l1)
        return out1 if f1_score(ref, out1, average='macro') >= f1_score(ref, out2, average='macro') else out2
        
        