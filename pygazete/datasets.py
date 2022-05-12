import os
import json
import jsonlines
import cv2 
import matplotlib.pyplot as plt
#plt.style.use('default')
import numpy as np
from IPython.display import display, HTML

from pygazete.preprocess import new_dir, allowed_format


############ PARSE JSON ##########################################################
def parse_json(ann: dict, keyword: str):
    assert keyword in ['tokens','bbox']
    cells = ann['html']['cells']
    result = [c[keyword] for c in cells if c['tokens']!=[] and 'bbox' in c.keys()]
    
    return result





############ TRIMAP GENERATION ####################################################
def generate_trimap(mask,eroision_iter=6,dilate_iter=8):
    """Takes a binary image 'mask' and returns a trimap"""
    d_kernel = np.ones((3,3))
    erode  = cv2.erode(mask,d_kernel,iterations=eroision_iter)
    dilate = cv2.dilate(mask,d_kernel,iterations=dilate_iter)
    unknown1 = cv2.bitwise_xor(erode,mask)
    unknown2 = cv2.bitwise_xor(dilate,mask)
    unknowns = cv2.add(unknown1,unknown2)
    unknowns[unknowns==255]=127
    trimap = cv2.add(mask,unknowns)

    labels = trimap.copy()
    labels[trimap==127]=1 #unknown
    labels[trimap==255]=2 #foreground
    
    return labels



def segmentation_mask(DIR, img_filename, bbox, debug = False):
    """Generates trimap masks for images from table cell coordinates, and saves binary
    cell masks and trimap masks in local folders
    
    arguments
    ---------------------
    DIR             str, path to input images
    img_filename    str, file name of input images
    bbox            array-like, table cell coordinates in a list
    debug           bool, whether to display debugging plots   
    
    returns
    ---------------------
    trimap          array-like, the generated trimap with labels (0,1,2)
                    0 for background
                    1 for unknown
                    2 for foreground
    
    
    
    """
    img_fn = os.path.join(DIR, img_filename)
    MASKDIR = new_dir(DIR, 'masks')
    TRIMAPDIR = new_dir(DIR, 'trimaps')
    
    img = cv2.imread(img_fn)
    mask = np.zeros(img.shape[:2],np.uint8)
    for x0,y0,x1,y1 in bbox:
        mask[y0:y1,x0:x1] = 255
    
    trimap = generate_trimap(mask)
    img_name = img_filename.split('.')[0]
    cv2.imwrite(os.path.join(MASKDIR, img_name+'_mask.png'),mask)
    cv2.imwrite(os.path.join(TRIMAPDIR, img_name+'_mask.png'),trimap)
    # debug
    if debug:
        plt.figure(figsize=(12,7))
        #imgplot = plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        imgplot = plt.imshow(trimap)
    

        
    return trimap