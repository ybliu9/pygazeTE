import cv2
import time
import os
import pytesseract as pt
from itertools import repeat
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pygazete.table_structure import *


############### debug #########################33
def show_images(raw, proc):
	# # Show image
	plt.figure(figsize=(12, 8))
	plt.title('Raw Image vs Processed Image')
	plt.subplot(2, 1, 1)
	plt.title('Raw')
	plt.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
	plt.subplot(2, 1, 2)
	plt.title('Processed')
	plt.imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.tight_layout()
	plt.show()

  
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
        



def extract(test_file_path, output_dir, mode=7, split='cell',borderless=True,display_data=False, debug=False):
	"""
    Reads in an image, performs table extraction, and saves prediction results in local files
    
    arguments
    -----------
    test_file_path      str, full path of input image
    output_dir          str, output directory to save output data
    mode                int, range [0,13] tesseract page segmentation mode
                        *Tip: mode = 6 reads in the whole table and returns texts in lines (parameter 'split' is ignored)
                              mode = 7 reads in a line or a cell. Please use split='cell' to reconstruct the table structure
                              mode = 12 reads in a line and returns lines with cells split by '|'
                        FOR THIS PROJECT, ONLY MODES 6,7,12 WERE TRAINED!
                        Reference: https://muthu.co/all-tesseract-ocr-options/
                        -- 0    Oritation and script detection (OSD) only
                        -- 1    Automatic page segmentation with OSD
                        -- 2    Automatic page segmentation, but no OSD or OCR
                        -- 3    Fully automatic page segmentation, but no OSD (Default)
                        -- 4    Assume a single column of text of variable sizes
                        -- 5    Assume a single uniform block of vertically aligned text   
                        -- 6    Assume a single uniform block of text
                        -- 7    Assume a single text line
                        -- 8    Assume a single word    
                        -- 9    Assume a single word in a circle
                        -- 10   Assume a single character
                        -- 11   Sparse text. Find as much text as possible in no particular order
                        -- 12   Sparse text with OSD
                        -- 13   Raw line. Assume a single text line, bypassing hacks that are Tesseract-specific
    split               str, ['row','cell'], the way to split the table, default is 'row'
                        When mode == 6, 'split' is ignored;
                        split = 'cell' with mode = 7 can reconstruct table structure but 
                        is twice slower than split = 'row'
    borderless          bool, whether the table is borderless
    display_data        bool, if display_data == True, display predicted data
    debug               bool, debug flag, if debug == True, display debugging plots
    
                        
    returns
    -----------
    data                str/DataFrame, table extraction results    
    
	"""
	assert mode in [6,7,12]
    
	image = cv2.imread(test_file_path)
	img_name = os.path.split(test_file_path)[-1].split('.')[0]
	raw = image.copy()
	
	# time
	t0 = time.time()
	if mode == 6:
		_, image = remove_background(image, method="binary")
		# OCR text recognition
		data = pt.image_to_string(image, lang='fintabnet_full', config='--psm '+str(mode))
		# save to .txt file
		output_path = os.path.join(output_dir, "%s_%s_psm_%s.txt"%(img_name, split, str(mode))) 
		with open(output_path, "w+") as f:
			f.write(data)
	else:
		image, coords = get_borders(img=image, plot=debug, method=split,
                                    borderless=borderless)
		dims, positions = box_positions(coords)
		# OCR text recognition by row/cell
		data = np.empty(dims, dtype=object)
		for (p,c) in zip(positions, coords):
			x,y,w,h = c
			i,j = p
			block = image[y:y+h,x:x+w]
			text = pt.image_to_string(block, lang='fintabnet_full', config='--psm '+str(mode))
			# remove '\n' (from mode 7) and replace '\n\n' (from mode 12)
			text = text.strip('\n').replace('\n\n','|').replace('—','-') if text else text
			data[i][j] = text
            
		data = reconstruction_check(pd.DataFrame(data))
		
		#save to .csv file
		output_path = os.path.join(output_dir, "%s_%s_psm_%s.csv"%(img_name, split, str(mode)))
		data.to_csv(output_path,encoding='UTF-8')
        
	t = time.time()-t0
	print("Time spent for processing and OCR: %.3f s"%t)

	# show preview for debugging
	if debug:   
		show_images(raw, image)
	if display_data:
		print(pd.DataFrame(data).to_markdown())
        
	return data
    
