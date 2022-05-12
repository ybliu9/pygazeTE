import cv2 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def remove_background(img=None, method='binary',thresh=150, tozero_thresh=[100,200]):
    """
    processes the input image by sharpening, thresholding, and recoloring, 
    and returns a binary/grayscale image
    
    arguments
    -----------
    img             array, the input RGB/RGBA image as a numpy array with shape (height, width, channel)
    method          str, method of image processing, e.g. 'binary', 'tozero', 'OTSU', 'adaptive'
                    -- 'binary' for binary thresholding --> generates binary image
                    -- 'tozero' for tozero thresholding --> generates grayscale image
                    -- 'OTSU' for OTSU thresholding --> generates binary image
                    -- 'adaptive' for adaptive thresholding --> generates binary image
    thresh          int, range [0,255], the threshold value for binary image, default = 150
    tozero_thresh   tuple/list (optional), the low and high thresholds applied ONLY when method='tozero', 
                    e.g. (100, 200) would change values under 100 to 0 (black) and change values 
                    over 200 to 255 (white), default = [100,200]
    
    
    returns
    -----------
    _params         str, a string containing input parameter values
    thresh_values   array, the processed grayscale image as a numpy array with shape (height, width)
    
    """
    # make sure the input image is valid
    assert img is not None and method in ['binary', 'tozero', 'OTSU', 'adaptive']

    
    # function parameters as constants
    sharpen = True
    
    #1. sharpening (not applied to adaptive shresholding)
    kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel) if method != 'adaptive' else img
    
    #rows, cols, channels = img.shape
    
    #2. grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #if channels >= 3 else img
    
    rows, cols = gray.shape
    
    #3. thresholding
    func = cv2.THRESH_BINARY#_INV 
    if method == 'binary':
        ret, thresh_values = cv2.threshold(gray,thresh, 255, func)
    elif method == 'tozero':
        ret, thresh_values = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY+cv2.THRESH_TOZERO)
        if tozero_thresh is not None and len(tozero_thresh) == 2 :
            l,h = tozero_thresh
            thresh_values = np.where(thresh_values >= h, 255, thresh_values)
            thresh_values = np.where(thresh_values <= l, 0, thresh_values)
    elif method == 'OTSU':
        ret, thresh_values = cv2.threshold(gray, thresh, 255, cv2.THRESH_OTSU)
    elif method == 'adaptive':
        thresh_values = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, func, 5, -1)

    
    #4. removing background: for tables with multiple background colors
    if check_background(thresh_values, axis=0) or check_background(thresh_values, axis=1):
        print('CHECK: The test image has split/contrasting background colors.')
        # Further process each row for images with multiple background colors
        for row in range(rows):
            # count number of white and black pixels of each row
            _nonzero, _zero = np.count_nonzero(thresh_values[row,:]), np.count_nonzero(thresh_values[row,:]==0)
        
            # if this row has no characters, skip
            if _zero == 0: continue
            # if this row has white characters in a dark background, flip the colors
            elif _zero > _nonzero or _nonzero == 0:
                # inverse colors
                thresh_values[row,:] = 255-thresh_values[row,:]
    
    #5. make sure output image has white background        
    _nonzero, _zero = np.count_nonzero(thresh_values), np.count_nonzero(thresh_values==0)
    if _zero > 2 * _nonzero:
        thresh_values = 255-thresh_values

    
    _params = method+', shreshold='+ str(thresh) + int(sharpen)*', sharpened'
    #return a parameter string and a binary/grayscale image with white background
    return _params, thresh_values
    

####################################################################
def contours(img, mask, axis):
    """generate contours for gridlines in image-form tables
    and returns a mask
    
    arguments
    -----------
    img             array, the input grayscale/binary image as a numpy array with 
                    shape (height, width)
    mask            array, an array of zeros with shape (height, width)
    axis            int, 0 or 1. axis = 1 detects rows, axis = 0 detects columns
    
    returns
    -----------
    mask            array, modified mask with white gridlines and black background
    """
    assert img is not None and axis in [0,1]
    
    dim = img.shape[axis]
    count = np.count_nonzero(img, axis=axis)
    # threshold count at width/height of image
    count_thresh = count.copy()
       
    #set lines without characters as white gridlines
    count_thresh[count>=dim-1] = 255
    #set all other lines to black
    count_thresh[count<dim-1] = 0

    # get contours
    count_thresh = count_thresh.astype(np.uint8)
    contours_ = cv2.findContours(count_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_[0] if len(contours_) == 2 else contours_[1]    
    
    
    rect = [cv2.boundingRect(cntr) for cntr in contours]
    #print("rect: ", rect)
    
    if axis == 1:
        # horizontal line coordinates
        coords = [r[1]+2*r[3]//3 for r in rect if r[3]>1]
        
    elif axis == 0:
        x,y,a,b = rect[0]
        # vertical lines coordinates
        coords = [r[1]+r[3]-2 for r in rect if r[3]>5]
    #    #coords = [r[1]+r[3]-3 if r[3] > 3 else r[1]+r[3] for r in rect]
        if b <= 5:
            coords = [rect[0][1]+5*rect[0][3]//6] + coords

    
    
    # reverse the order
    coords.reverse()
    coords.append(img.shape[1-axis]-1)
    #print("coords: ",coords)
    
    dif = list(np.diff(coords))
    dif_ = sorted(dif)[:-1]
    dif = [coords[0]] + dif
    #print("dif: ", dif)
    #print(len(coords), len(dif))
    
    _mean = np.mean(dif_)
    if axis == 1:
        idx = list(np.where(dif >= 0.8 *_mean)[0])
    if axis == 0:
        idx = list(np.where(dif >= 0.85* _mean)[0])
    #print(idx)
    trim_c = [coords[0]]
    trim_c.extend(np.array(coords)[idx])
    
    for c in trim_c:
        if axis == 0:
            cv2.line(mask, (c,0), (c,dim-1), (255, 0, 0), 2)
        elif axis == 1:
            cv2.line(mask, (0,c), (dim-1,c), (255, 0, 0), 2)
    
    return mask   
    
####################################################    
def get_borders(img=None, method = 'row', borderless=True, 
                erode0_iter=1, erode1_iter=1, dilate0_iter=3, dilate1_iter=3,
                plot=False, img_name=None):
    """
    gets structures of an image-format table and returns coordinates of cells/rows within the table
    
    arguments
    -----------
    img             str/array, the path to image (string), or the input RGB/RGBA image (array) as a numpy
                    array with shape (height, width, channel)
    method          str, ('cell', 'row'), returns coordinates of all cells if method == 'cell', otherwise
                    returns coordinates of all rows
    borderless      bool, whether the table is borderless, default = True
    erode0_iter     int, range [1,inf), the number of iterations for horizontal erosion, default = 1
    erode1_iter     int, range [1,inf), the number of iterations for vertical erosion, default = 1
    dilate0_iter    int, range [1,inf), the number of iterations for horizontal dilation, default = 3
    dilate1_iter    int, range [1,inf), the number of iterations for vertical dilation, default = 3
    plot            bool, display results if plot = True, default = False
    img_name        str, name of the image displayed in plots (if plot == True)
    
    returns
    -----------
    coords          array, the processed image with structural lines, as a numpy array with 
                    shape (height, width)
    
    """
    
    # make sure the input image is valid
    assert img is not None and method in ['cell','row']
    
    
    if type(img) == str:
        s = '/' if img.find('/') != -1 else '\\'
        img_name = img.split(s)[-1].split('.')[0]
        img = cv2.imread(img)
    else:
        assert len(img.shape) == 3
    
    if borderless:
        _, thresh = remove_background(img, method='binary')
        dims = thresh.shape
    
        # count number of non-zero pixels in each row/column
        mask = np.zeros(dims, np.uint8)
        line_indices = _incomplete_lines(thresh)
        # for row detection, remove all gridlines
        processed = remove_incomplete_lines(thresh,line_indices, axis=1)
        mask = contours(processed, mask, axis = 1)
        # for column detection, only remove long horizontal gridlines for more accuracy recognition
        mask = contours(remove_incomplete_lines(thresh,line_indices, axis=0), mask, axis = 0)

    else:
        thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = 255-thresh
        processed = thresh.copy()
        
    r,c = mask.shape
    
    k0, k1 = int(np.sqrt(c)*1.2), int(np.sqrt(r)*1.2)
    
    #recognize horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k1,1))
    eroded = cv2.erode(mask, kernel, iterations = erode1_iter)
    #change iterations based on image quality
    dilatedrow = cv2.dilate(eroded, kernel, iterations = dilate1_iter)
    
    
    
    if method == 'cell':
        #recognize vertical lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,k0))
        eroded = cv2.erode(mask, kernel, iterations = erode0_iter)
        dilatedcol = cv2.dilate(eroded, kernel, iterations = dilate0_iter)
    elif method == 'row':
        dilatedcol = np.zeros((r,c), np.uint8)
        dilatedcol[:,0] == 255
        dilatedcol[:,c-1] == 255
    
    #recognize intersections
    bitwiseAnd = cv2.bitwise_and(dilatedcol,dilatedrow)
    

    #merge
    merge = cv2.add(dilatedcol, dilatedrow)
    _, merge_c = cv2.threshold(merge, 50, 255, cv2.THRESH_BINARY)
    
    #get intersection coordinates
    xs, ys = np.where(bitwiseAnd>1)
    
    xs.sort()
    ys.sort()
    #print(np.unique(xs))
    #print(np.unique(ys))
    
    
    contours_, hierarchy = cv2.findContours(255-merge_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours_[0] if len(contours_) == 2 else contours_[1] 
    boxes = [cv2.boundingRect(c) for c in contours_]
    
    
    # remove narrow boxes with width less than 1/40 of rows or height less than 1/50 of columns
    boxes = [b for b in boxes if b[2]>=c//40 and b[3]>=r//40]
    coords = sort_boxes(boxes)


    
    if plot: 
        # show intersection points more clearly
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        bitwiseAnd = cv2.dilate(bitwiseAnd, kernel, iterations = 1)
        
        im = img.copy()
        for (x,y,w,h) in coords:
            cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
        
        
        plots_ = dict()
        
        plots_['original image with border coordinates'] = im
        plots_['thresholding'] = thresh
        plots_['remove lines (column)'] = remove_incomplete_lines(thresh,line_indices, axis=0)
        plots_['remove lines (row)'] = remove_incomplete_lines(thresh,line_indices, axis=1)
        plots_['binary mask'] = mask
        subtitle = img_name + ': ' if img_name is not None else ""
        if method == 'cell':
            plots_['intersections'] = bitwiseAnd
            plots_['merged'] = merge_c
            plots_['dilated columns'] = dilatedcol
        plots_['dilated rows'] = dilatedrow
        
        ht = max(5, r//100)
        fig, axes = plt.subplots(len(plots_), 1, sharex=False, #True,
                         gridspec_kw=(dict(hspace=0.3,wspace=0.1)), 
                         figsize=(15,ht*len(plots_))
                         )
        d_ = list(plots_.items())
        for i in range(len(plots_)):
            axes[i].imshow(cv2.cvtColor(d_[i][1], cv2.COLOR_BGR2RGB))
            axes[i].set_title(subtitle + d_[i][0])

        #axes[0].scatter(coords.T[1], coords.T[0], s=15, c='r')
        plt.show()

    
    return processed, coords   


######################################################################
# helper functions
def check_background(binary, axis):
    """
    check if rows/columns of an image have dark background and light texts 
    
    
    arguments
    -----------------
    binary      array, a binary image
    axis        int, 0 or 1, axis=1 for rows and axis=0 for columns
    
    returns
    -----------------
    (bool)      returns True if rows/columns have split background colors, False otherwise
    
    """
    rows, cols = binary.shape
    # count zero values for all rows/columns
    zeros = np.count_nonzero(binary==0, axis=axis)
    # possible rows/columns with dark background and light texts
    _zero = np.where(zeros > binary.shape[1-axis]//2)[0]
    # if <=10 pixels of rows/columns have dark background, return False
    if len(_zero) <= 10:
        return False
    # if consecutive 10 rows/columns have dark background and light texts, return True
    else:
        dif = np.diff(_zero)
        for i in range(10,len(dif),1):
            if all(dif[i-10:i] == 1):
                return True
        return False
        
        
def _incomplete_lines(thresh_values):   
    """
    find row/column index for incomplete gridlines
    """
    rows, cols = thresh_values.shape
    
    ver_lines = []
    #remove vertical lines
    for c in range(cols):
        for r in range(rows//4, rows, 5):
            # if the column has a semi-structured/incomplete vertical line with length of at
            # least 1/4 of the line, complete the line 
            if all(thresh_values[(r-rows//4):r,c] < 20):
                ver_lines.append(c)
                break
    #remove horizontal lines
    hor_lines = []
    for r in range(rows):
        for c in range(cols//25, cols, 2):
            # if the row has a semi-structured/incomplete horizontal line with length of at
            # least 1/25 of the width, complete the line 
            if all(thresh_values[r,(c-cols//25):c] < 20):
                hor_lines.append(r)
                break
    return ver_lines, hor_lines



def remove_incomplete_lines(thresh_values, line_indices, axis):   
    """
    remove incomplete gridlines
    """
    th_copy = thresh_values.copy()
    ver_lines, hor_lines = line_indices
    
    if axis == 1:
        th_copy[hor_lines,:] = 255
        th_copy[:,ver_lines] = 255
    elif axis == 0:
        for r in hor_lines:
            _r = np.argwhere(thresh_values[r,:]<20)
            # remove long lines but keep short lines
            a = np.where(np.diff(_r.reshape(-1,)!=1, 0, 1))
            if np.count_nonzero(a==0)==0 or max_consecutive(a) > thresh_values.shape[1]//2:
                th_copy[r,:] = 255

    return th_copy
    

        
def sort_boxes(boxes):
    """Sort boxes coordinates from top to bottom, left to right"""
    boxes.sort(key = lambda b: (b[1],b[0]))
    return boxes
    

def box_positions(boxes):
    """Reads a list of box coordinates and calculate each box's position in the table
    Returns dimensions of the table and cell positions 
    
    arguments
    -----------
    boxes           array, boxes recognized in a table as a list of tuples, each tuple containing 
                    the coordinates of a box

    returns
    -----------
    dims            tuple, (rows, cols), the maximum number of rows and cols recognized in the table
    positions       array, the processed grayscale image as a numpy array with shape (height, width)
    """
    print(boxes)
    col_idx = list(np.unique(np.array(boxes)[:,0]))
    row_idx = list(np.unique(np.array(boxes)[:,1]))
    max_cols = len(col_idx)
    max_rows = len(row_idx)
    dims = (max_rows, max_cols)
    
    bboxes = np.array(boxes)
    positions = []
    for i in range(bboxes.shape[0]):
        cidx = col_idx.index(bboxes[i][0])
        ridx = row_idx.index(bboxes[i][1])
        #print(cidx,ridx)
        positions.append([ridx,cidx])
    
    assert len(boxes) == len(positions)
    
    return dims, positions
    
def max_consecutive(a:np.array):
    """ count maximum number of consecutive repeated values """
    a_ext = np.concatenate(( [0], a, [0] ))
    idx = np.flatnonzero(a_ext[1:] != a_ext[:-1])
    a_ext[1:][idx[1::2]] = idx[1::2]-idx[::2] 
    return max(a_ext)
    
def allEmpty(a: np.array):
    return all(a == '')

def reconstruction_check(df:pd.DataFrame):
    """check if a reconstructed table have empty rows/columns and remove them if they exist"""
    #check columns
    check = df.apply(allEmpty, axis=0)
    #drop empty columns
    if any(check==True):
        df = df.drop(check[check==True].index[0], axis=1)
        #reset columns indices
        df.columns = range(df.columns.size)
    
    #check rows
    check = df.apply(allEmpty, axis=1)
    #drop empty columns
    if any(check==True):
        df = df.drop(check[check==True].index[0], axis=0).reset_index(drop=True)
    
    return df
        