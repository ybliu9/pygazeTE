# Opengaze Data Preprocessing
#
# Preprocess opengaze data and log for analytic purpose.
#
# (6 April 2022)


############ NOT FINISHED ###################

__author__ = "Chloe Liu"

# native
import os
# external
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pygazete.constants import *



def process_data(datadir:str, datafile:str, logfile:str, ntrials:int):
    """ Read data from files, remove calibration data points and unwanted columns, 
        and return processed data as a pandas dataframe.
        
    Arguments: 
    datadir       string, the directory of the data and log files
    datafile      string, the name of the data file
    logfile       string, the name of the log file
    ntrials       integer, the number of images displayed in each experiment
    
    Returns:
    df            dataframe, the processed data
    
    """
    if datafile is None or logfile is None: 
        raise ValueError("Invalid data file or log file!")
    DATAPATH = os.path.join(datadir, datafile)
    LOGPATH = os.path.join(datadir, logfile)
    # read data
    df = pd.read_csv(DATAPATH, sep='\t')    
    # read log
    with open(LOGPATH,'r') as f:
        log = f.read().splitlines()         #.readlines()
        f.close()
        #f.close()
    # trim log
    _s = log.index('pygaze calibration report end')+1
    _e = log.index('pygaze experiment report end')
    log_trim = [c.upper() for c in log[_s:_e]]
    
    # Find index where the first trial starts (remove rows recorded during calibration)
    i = df.USER.str.startswith('TRIALSTART').idxmax()
    # Select columns
    COLUMNS = list(range(15))+list(range(18,28))+list(range(len(df.columns)-1, len(df.columns)))
    
    # Slicing
    df = df.iloc[i:, COLUMNS].reset_index(drop=True)
    
    # create new columns to store image-specific info
    
    df['RESCALE'] = None
    df['IMGSIZE'] = None
    df['IMGID'] = -1
    
    
    for i in range(ntrials):
        s = df.index[df.USER.isin(log_trim[i*7:i*7+5])].tolist()
        e = df.index[df.USER.isin(log_trim[i*7+5:(i+1)*7])].tolist()
        if s and e:
            r_idx = 
            df.iloc[s[0]:e[-1]+1, -1] = log_trim[i*7+1][10:]#.lstrip('IMAGENAME ')
            df.iloc[s[0]:e[-1]+1, -2] = log_trim[i*7+2][8:]
            df.iloc[s[0]:e[-1]+1, -3] = float(log_trim[i*7+3][8:])
        else:
            #print(s,e,log_trim[i*5+1][10:])
            continue
    
    return df
 
 
def _split(obj,i):
    s = str(obj)[1:-1].split(',')
    return s[i]

def trim_edge(data=None, dur_thres=0.2, edge = 10, fix_flag=True, filename=None, DISPSIZE=DISPSIZE):
    """ remove points of gaze at the screen edges because 
        they are not likely to contain useful information.
    
    Arguments:
    data         DataFrame, data collected from a single trial
    dur_thres    float, duration threshold for fixation, in seconds
    edge         integer, trimed edge in pixels
    filename     string (optional), full path to the file in which the trimmed data should be
                 saved, or None to not save the file (default = None)
    DISPSIZE     tuple, size of display in pixels
    
    Returns:
    data         DataFrame, trimmed data
    
    """
    if data is None:
        raise ValueError("Invalid argument! 'data' must be a non-empty data frame containing fixation datapoints.")
    
    data = data[data.FPOGD >= dur_thres]
    if fix_flag:
        data = data[data.FPOGV == 1]
    data['EDGEX'] = (DISPSIZE[0] - data['IMGSIZE'].apply(_split,args=(0,)).astype('int32')*data['RESCALE'])//2 + edge
    data['EDGEY'] = (DISPSIZE[1] - data['IMGSIZE'].apply(_split,args=(1,)).astype('int32')*data['RESCALE'])//2 + edge
    data['FPOGXp'] = data['FPOGX']* DISPSIZE[0]
    data['FPOGYp'] = data['FPOGY']* DISPSIZE[1]
    data = data[data['FPOGXp']>data['EDGEX']]
    data = data[data['FPOGXp']<DISPSIZE[0]-data['EDGEX']]
    data = data[data['FPOGYp']>data['EDGEY']]
    data = data[data['FPOGYp']<DISPSIZE[1]-data['EDGEY']]
    # reset (0,0)
    data['FPOGXp'] =(data['FPOGXp'] - data['EDGEX'] + edge)/data['RESCALE']
    data['FPOGYp'] = (data['FPOGYp'] - data['EDGEY'] + edge)/data['RESCALE']
    
    if filename:
        data.to_csv(filename)
    return data.reset_index(drop=True)
    

def pred_coords(predlabdir:str):
    """Extract the predicted table coordinates from the directory
    
    Arguments: 
    predlabdir    string, the directory of predicted labels of the tables
    IMGSIZE       tuple, size of image in pixels
    
    Returns:
    coords        dataframe consists of three columns: img_name, boxes, and box_pixels.
                  - 'boxes' contains the x-center, y-center, width, and height of all the tables
                  within the image as percentages of the image width and height
                  - 'box_pixels' contains the x-min, y-min, x-max, and y-max of all the tables
                  within the image as pixels of the image                 
    
    """
    lb = os.listdir(predlabdir)
    coords = pd.DataFrame(columns=('img_name', 'boxes', 'box_pixels'))
    for path in lb:

        if path.endswith('.txt'):
            boxes = np.loadtxt(os.path.join(predlabdir,path)).reshape((-1,5))
        
            box, box_pixels = [], []
            for i,b in enumerate(boxes):
                # x, y, w, h
                b = b[1:]
                box.append(list(b))
                box_pixels.append([(b[0]-b[2]/2)*IMGSIZE[0], (b[1]-b[3]/2)*IMGSIZE[1], 
                                   (b[0]+b[2]/2)*IMGSIZE[0]+2, (b[1]+b[3]/2)*IMGSIZE[1]])
            row = {'img_name': path.strip('.txt'), 'boxes': box, 'box_pixels':box_pixels}
            #print(row)
            coords = coords.append(row, ignore_index = True)
    return coords


def count_density(data=None, boxes=None, imglist=None):
    """
    Count density in the bounding boxes of each table, as a percentage.
    
    Arguments:
    data         dataFrame, cleaned and trimmed data
    boxes        dataFrame, x and y coordinates of the bounding boxes
    imglist      list, containing image names. To avoid ValueError, imglist must be a non-empty 
                 list (default = None) 
    
    Returns:
    m            a defaultdict using list, with key as the image name and value as a
                 list of fixation densities for all tables in the image
    
    """
    
    if data is None:
        raise ValueError("Invalid argument! 'data' must be a non-empty data frame containing fixation datapoints.")
    if boxes is None:
        raise ValueError("Invalid argument! 'boxes' must be a non-empty data frame containing predicted table coordinates.")
    if imglist is None:
        raise ValueError("Invalid argument! 'imglist' must be a non-empty list containing image names.")
    z = len(imglist)
    m = defaultdict(list)
    for img in imglist:
        img_name = img.upper()
        datai = data[data.IMGID==img_name].reset_index(drop=True)
        RESCALE = datai['RESCALE'][0]
        _len = len(datai)
        if _len == 0: m[img] = []
        else:
            imgi = img.lower().split('.')[0]
            boxi = boxes[boxes.img_name == imgi].box_pixels.values[0]
            dens_list = []
            for box in boxi:
                #xmin, ymin, wd, ht = box
                xmin, ymin, xmax, ymax = [int(i * RESCALE) for i in box]
                #calculate density in percentage
                density = round(len(datai[(datai.FPOGXp>=xmin-10)&(datai.FPOGXp<=xmax+10) & 
                                          (datai.FPOGYp>=ymin-10)&(datai.FPOGYp<=ymax+10)])/_len * 100,2)
                dens_list.append(density)
            m[img] = dens_list
    return m


def dbscan(data=None, eps=50, min_samples=5):
    """
    Perform DBSCAN clustering to extract coordinates for bounding boxes.
    
    Arguments:
    data         dataFrame, cleaned and trimmed data
    eps          int/float, same model parameter as sklearn.cluster.DBSCAN (default = 50)
    min_samples  int, same model parameter as sklearn.cluster.DBSCAN (default = 5)
    
    Returns:
    box          a defaultdict using list, with key as the image name and value as a
                 list of coordinates in pixels for AOI   
    
    """
    centroids = defaultdict(list)
    for img in list(pd.unique(data.IMGID)):
        X = data[data.IMGID == img][['FPOGXp','FPOGYp']].to_numpy()

        model = DBSCAN(eps=eps,min_samples=min_samples, n_jobs=-1) # 
        yhat = model.fit_predict(X)
    
        # retrieve unique clusters
        clusters = np.unique(yhat)
        labels = model.labels_
    
        coord = []
        for cluster in clusters:
            if cluster == -1:continue
            else:
                centroid = np.median(X[labels==cluster,:], axis=0)
                coord.append(list(centroid))
        centroids[img] = coord
        
    box = defaultdict(list)
    for k,v in centroids.items():
        a = np.array(v).T
        #print(a)
        xmin, xmax = np.min(a[0], axis=0), np.max(a[0], axis=0)
        ymin, ymax = np.min(a[1], axis=0), np.max(a[1], axis=0)
        #print(xmin, xmax, ymin, ymax)
        x = (xmin+xmax)/2
        y = (ymin+ymax)/2
        w = xmax-xmin
        h = ymax-ymin
        box[k.lower()] = [round(x), round(y), round(w), round(h)]
    return box
    
    
def new_dir(parent, new):
    # make new directory 'new' under 'parent' if not exists
    path = os.path.join(parent, new)
    if not os.path.exists(path):
        os.mkdir(path)
    return path
    
def allowed_format(f):
    # check image format
    allowed = ['.png','.jpg','.jpeg','.PNG','.JPG','.JPEG']
    return any(f.endswith(e) for e in allowed)