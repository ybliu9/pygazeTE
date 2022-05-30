# -*- coding: utf-8 -*-

# Pygazeplot is a simplified version of the Python module PyGazeAnalyser 
# v-0.1.0 Copyright (C) 2014 created by Edwin S. Dalmaijer for easily 
# analysing eye-tracking data. 
#
#   Pygazeplot is modified by Chloe Liu for creating visualizations of 
#   eye-tracking log data generated from PyGaze experiments.
#

# Gaze Plotter
#
# Produces different kinds of plots that are generally used in eye movement
# research, e.g. heatmaps, scanpaths, and fixation locations as overlays of
# images.
#
# (4 April 2022)

__author__ = "Chloe Liu"

# native
import os
# external
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')
from matplotlib import image, cm, colors
from PIL import Image
import seaborn as sns
from pygazete.constants import *
sns.set(rc = {'figure.figsize':(18,18)})

# # # # #
# SETTINGS

# COLORS
# all colors are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
COLS = {	"butter": [	'#fce94f',
					'#edd400',
					'#c4a000'],
		"orange": [	'#fcaf3e',
					'#f57900',
					'#ce5c00'],
		"chocolate": [	'#e9b96e',
					'#c17d11',
					'#8f5902'],
		"chameleon": [	'#8ae234',
					'#73d216',
					'#4e9a06'],
		"skyblue": [	'#729fcf',
					'#3465a4',
					'#204a87'],
		"plum": 	[	'#ad7fa8',
					'#75507b',
					'#5c3566'],
		"scarletred":[	'#ef2929',
					'#cc0000',
					'#a40000'],
		"aluminium": [	'#eeeeec',
					'#d3d7cf',
					'#babdb6',
					'#888a85',
					'#555753',
					'#2e3436'],
		}


# # # # #
# FUNCTIONS


def draw_fixations(data, imagefile=None, durationsize=True, durationcolour=True, alpha=0.5, savefilename=None, returnflag = False):
	
	"""Draws circles on the fixation locations, optionally on top of an image,
	with optional weigthing of the duration for circle size and colour
	
	arguments
	
	data            -   log data generated from a single trial, as a dataframe
    
	
	keyword arguments
	
	imagefile		-	full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)
	durationsize	-	Boolean indicating whether the fixation duration is
					to be taken into account as a weight for the circle
					size; longer duration = bigger (default = True)
	durationcolour	-	Boolean indicating whether the fixation duration is
					to be taken into account as a weight for the circle
					colour; longer duration = hotter (default = True)
	alpha		    -	float between 0 and 1, indicating the transparancy of
					the heatmap, where 0 is completely transparant and 1
					is completely untransparant (default = 0.5)
	savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)
    returnflag      -   boolean variable indicating whether to return the figure
                    (default = False)
	
	returns
	
	fig			    -	a matplotlib.pyplot Figure instance, containing the
					fixations if returnflag == True
	"""
	# FIXATIONS
	fix = parse_fixations(data)
    
	
	# IMAGE
	fig, ax = draw_display(DISPSIZE, imagefile=imagefile)

	# CIRCLES
	# duration weights
	if durationsize:
		siz = 1 * (fix['dur']*1000/5)
	else:
		siz = 1 * np.median(fix['dur']*1000/5)
	if durationcolour:
		col = fix['dur']
	else:
		col = COLS['chameleon'][2]
	# draw circles
	ax.scatter(fix['x'],fix['y'], s=siz, c=col, marker='o', cmap='jet', alpha=alpha, edgecolors='none')

	# FINISH PLOT
	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)
	
	if returnflag == True: return fig
	else: return

def draw_heatmap(data, imagefile=None, durationweight=True, alpha=0.5, savefilename=None, returnflag=False):
	
	"""Draws a heatmap of the provided fixations, optionally drawn over an
	image, and optionally allocating more weight to fixations with a higher
	duration.
	
	arguments
	
	data            -   data from a single trial, as a dataframe

	
	keyword arguments
	
	imagefile		full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)
	durationweight	-	Boolean indicating whether the fixation duration is
					to be taken into account as a weight for the heatmap
					intensity; longer duration = hotter (default = True)
	alpha		    -	float between 0 and 1, indicating the transparancy of
					the heatmap, where 0 is completely transparant and 1
					is completely untransparant (default = 0.5)
	savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)
    returnflag      -   boolean variable indicating whether to return the figure
                    (default = False)
	
	returns
	
	fig			    -	a matplotlib.pyplot Figure instance, containing the
					heatmap, if returnflag == True
	"""

	# FIXATIONS
	fix = parse_fixations(data)
	
	# IMAGE
	fig, ax = draw_display(DISPSIZE, imagefile=imagefile)

	# HEATMAP
	dispsize = DISPSIZE
	# Gaussian
	gwh = 200
	gsdwh = gwh/6
	gaus = gaussian(gwh,gsdwh)
	# matrix of zeroes
	strt = gwh//2
	heatmapsize = dispsize[1] + 2*strt, dispsize[0] + 2*strt
	heatmap = np.zeros(heatmapsize, dtype=float)
	# create heatmap
	for i in range(len(fix['dur'])):
		# get x and y coordinates
		#x and y - indexes of heatmap array. must be integers
		x = strt + int(fix['x'][i]) - int(gwh/2)
		y = strt + int(fix['y'][i]) - int(gwh/2)
		# correct Gaussian size if either coordinate falls outside of
		# display boundaries
		if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
			hadj=[0,gwh];vadj=[0,gwh]
			if 0 > x:
				hadj[0] = abs(x)
				x = 0
			elif dispsize[0] < x:
				hadj[1] = gwh - int(x-dispsize[0])
			if 0 > y:
				vadj[0] = abs(y)
				y = 0
			elif dispsize[1] < y:
				vadj[1] = gwh - int(y-dispsize[1])
			# add adjusted Gaussian to the current heatmap
			try:
				heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * fix['dur'][i]
			except:
				# fixation was probably outside of display
				pass
		else:				
			# add Gaussian to the current heatmap
			heatmap[y:y+gwh,x:x+gwh] += gaus * fix['dur'][i]
	# resize heatmap
	heatmap = heatmap[strt:dispsize[1]+strt,strt:dispsize[0]+strt]
	# remove zeros
	lowbound = np.mean(heatmap[heatmap>0])
	heatmap[heatmap<lowbound] = np.NaN
	# draw heatmap on top of image
	ax.imshow(heatmap, cmap='jet', alpha=alpha)

	# FINISH PLOT
	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)
	
	if returnflag == True: return fig
	else: return


#plot probabilities
def draw_density_heatmap(data, savefilename=None, returnflag=False):
    """Draws the density heatmap 

    arguments
    
    data            -   density data, as a 3-d array
    savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)
    returnflag      -   boolean variable indicating whether to return the figure
                    (default = False)
	
	returns
	
	fig			    -	a matplotlib.pyplot Figure instance, containing the
					density heatmap, if returnflag == True
    """
    plt.clf()
    fig = sns.heatmap(data.T, annot=True, fmt=".1f", cbar_kws={'label': 'Probability (%)'})

    if savefilename != None:
        plt.savefig(savefilename, dpi=200)
        
    
    if returnflag == True: return fig
    else: return
    

def draw_raw(x, y, imagefile=None, savefilename=None, returnflag=False):
	
	"""Draws the raw x and y data
	
	arguments
	
	x			-	a list of x coordinates of all samples that are to
					be plotted
	y			-	a list of y coordinates of all samples that are to
					be plotted

	imagefile		-	full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)
	savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)
    returnflag      -   boolean variable indicating whether to return the figure
                    (default = False)
	
	returns
	
	fig			    -	a matplotlib.pyplot Figure instance, containing the
					fixations, if returnflag == True
	"""
	
	# image
	fig, ax = draw_display(DISPSIZE, imagefile=imagefile)

	# plot raw data points
	ax.plot(x, y, 'o', color=COLS['aluminium'][0], markeredgecolor=COLS['aluminium'][5])

	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)
	
	if returnflag == True: return fig
	else: return


def draw_scanpath(data, imagefile=None, alpha=0.5, savefilename=None, returnflag=False):
    
    """Draws a scanpath: a series of arrows between numbered fixations,
    optionally drawn over an image
    arguments
    
    data		    -	log data generated from a single trial, as a dataframe    
    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)
    returnflag      -   boolean variable indicating whether to return the figure
                    (default = False)
    
    returns
    
    fig			    -	a matplotlib.pyplot Figure instance, containing the
                    heatmap, if returnflag == True
    """
    
    # image
    fig, ax = draw_display(DISPSIZE, imagefile=imagefile)

    # FIXATIONS
    # parse fixations
    fix = parse_fixations(data)
    # parse saccades
    sac = parse_saccades(data)
    
    # draw fixations
    ax.scatter(fix['x'],fix['y'], s=(1 * fix['dur'] / 30.0), c=COLS['chameleon'][2], marker='o', cmap='jet', alpha=alpha, edgecolors='none')
    #ax.scatter(fix['x'][0],fix['y'][0], s=(1 * fix['dur'] / 30.0), c=COLS['skyblue'][0], marker='o', cmap='jet', alpha=alpha, edgecolors=COLS['skyblue'][0])
    #ax.scatter(fix['x'][-1],fix['y'][-1], s=(1 * fix['dur'] / 30.0), c=COLS['orange'][0], marker='o', cmap='jet', alpha=alpha, edgecolors=COLS['orange'][0])
    # draw annotations (fixation numbers)
    
    cmap = cm.get_cmap('jet', len(data))
    clist = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    
    for i in range(len(fix)):
        ax.annotate(str(i+1), (fix['x'][i],fix['y'][i]), color=clist[i], size = 5, alpha=alpha, horizontalalignment='center', verticalalignment='center', multialignment='center')
        #COLS['aluminium'][5]

    # SACCADES
    # loop through all saccades
    for sx, sy, ex, ey in sac:
        # draw an arrow between every saccade start and ending
        ax.arrow(sx, sy, ex-sx, ey-sy, alpha=alpha, fc=COLS['aluminium'][0], ec=COLS['aluminium'][5], fill=True, shape='full', width=8, head_width=10, head_starts_at_zero=False, overhang=0)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)
    
    if returnflag == True: return fig
    else: return


# # # # #
# HELPER FUNCTIONS


def draw_display(dispsize, imagefile=None):
    
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it
    
    arguments
    
    dispsize            tuple or list indicating the size of the display,
                        e.g. (1024,768)
    
    keyword arguments
    
    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    
    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """
    # construct screen (black background)
    _, ext = os.path.splitext(imagefile)
    ext = ext.lower()
    data_type = 'float32' if ext == '.png' else 'uint8'
    #screen = np.zeros((h,w,3), dtype=data_type)
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        #img = image.imread(imagefile)

        img = Image.open(imagefile)
        # width and height of the image
        w, h = img.size
        #img = img.resize((int(w*RESCALE,int(h*RESCALE)))
        img = np.asarray(img,dtype=np.float32)/255
        screen = np.zeros((h,w,img.shape[2]), dtype=data_type)
        # flip image over the horizontal axis
        # (do not do so on Windows, as the image appears to be loaded with
        # the correct side up there; what's up with that? :/)
        if not os.name == 'nt':
            img = np.flipud(img)
        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        #x = dispsize[0]//2 - w//2
        #y = dispsize[1]//2 - h//2
        # draw the image on the screen
        #screen[y:y+h,x:x+w,:] += img
        screen[:h, :w, :] += img
    # dots per inch
    dpi = 200.0
    # determine the figure size in inches
    figsize = (w//dpi, h//dpi)
    # create a figure
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0,w,0,h])
    ax.imshow(screen)#, origin='upper')
    
    return fig, ax


def gaussian(x, sx, y=None, sy=None):
	
	"""Returns an array of numpy arrays (a matrix) containing values between
	1 and 0 in a 2D Gaussian distribution
	
	arguments
	x		-- width in pixels
	sx		-- width standard deviation
	
	keyword argments
	y		-- height in pixels (default = x)
	sy		-- height standard deviation (default = sx)
	"""
	
	# square Gaussian if only x values are passed
	if y == None:
		y = x
	if sy == None:
		sy = sx
	# centers	
	xo = x/2
	yo = y/2
	# matrix of zeros
	M = np.zeros([y,x],dtype=float)
	# gaussian matrix
	for i in range(x):
		for j in range(y):
			M[j,i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy)) ) )

	return M


def parse_fixations(data):
    """Returns all relevant data from a list of fixation ending events
	arguments
	data		log data from a single trial, as a dataframe
    
	returns
	fix		-	a dict with three keys: 'x', 'y', and 'dur' (each contain
				a numpy array) for the x and y coordinates and duration of
				each fixation
	"""
    data = data[data.FPOGV==1]
    fix = {	'x':np.array(data.loc[:, 'FPOGXp']),
			'y':np.array(data.loc[:, 'FPOGYp']),
			'dur':np.array(data.loc[:, 'FPOGD'])}
    return fix



def parse_saccades(data):
    """Returns all relevant data from a list of fixation events
	arguments
	data		log data from a single trial, as a dataframe
    
	returns
	sac		-	a numpy array with four columns: 'sx', 'sy', 'ex', and 'ey' 
				for the starting and ending positions of the x and y coordinates 
	"""
    data = data[data.FPOGV==1]
    a,b = data[1:], data[-1:]
    end = pd.concat([a,b],ignore_index=True)
    end = end.loc[:, ['FPOGXp','FPOGYp']]
    
    start = data.loc[:, ['FPOGXp','FPOGYp']].reset_index(drop=True)
    
    sac = pd.concat([start,end], ignore_index=True, axis = 1)
    return np.array(sac)
