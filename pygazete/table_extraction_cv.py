# import packages
import os
import random
import time
from collections import defaultdict
import keyboard
import pylink

from pygaze.libscreen import Display, Screen
from pygaze.libinput import Keyboard
from pygaze.eyetracker import EyeTracker
from pygaze.liblog import Logfile
import pygaze.libtime as timer
from pygaze.mouse import Mouse

import numpy as np
import pandas as pd


# Self-built package
from pygazete.gazeplotter import *
from pygazete.preprocess import *
from pygazete.constants import *
from pygazete.table_structure import *
from pygazete.ocr import *


# ignore warnings
import warnings
warnings.filterwarnings('ignore')




def advanced(RESCALE=RESCALE, MAXTRIALTIME=MAXTRIALTIME,
             DISPSIZE=DISPSIZE, IMGSIZE=IMGSIZE, ip=ip, **kwargs):
    try:
        kwargs['DIR'] and os.path.exists(kwargs['DIR'])
        DIR = kwargs['DIR']
    except:
        DIR = os.getcwd()
        print("Warning: the path keyword 'DIR' is not found.\nData would be saved under default directory: %s" % DIR)
    
    try:
        kwargs['IMGDIR'] and os.path.exists(kwargs['IMGDIR'])
        IMGDIR = kwargs['IMGDIR']
    except KeyError:
        raise KeyError("The path keyword 'IMGDIR' is invalid or not found! Please pass the correct directory of test images")
    
    # set directories
    RESULTDIR = os.path.join(DIR, 'runs\\detect')
    tstamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
    
    exps = os.listdir(RESULTDIR)
    exps = sorted([e for e in exps if e != 'exp' and e.startswith('exp')], key = lambda x: int(x[3:]))
    PREDDIR = os.path.join(RESULTDIR, exps[-1])
    LB_PREDDIR = os.path.join(PREDDIR, 'labels')
    
    PLOTDIR = new_dir(PREDDIR, 'plots')
    DATADIR = new_dir(PREDDIR, 'gaze_data')
    LOGFILENAME = tstamp+'D'+str(MAXTRIALTIME//1000)
    LOGFILE = os.path.join(DATADIR, LOGFILENAME)
    
    # read test images    
    images = [f for f in os.listdir(IMGDIR) if allowed_format(f)]
    ntrials = len(images)
    
    # read predicted table coordinates
    coords = pred_coords(LB_PREDDIR)
    

    # initialize objects
    disp = Display()
    scr = Screen()

    mouse = Mouse(mousebuttonlist=None, timeout=10000)
    mouse.set_visible(visible=True)

    # create keyboard object
    kb = Keyboard(keylist=['space'], timeout=10000)
    init_t0 = time.time()
    tracker = EyeTracker(disp, debug=True, logfile=LOGFILE)

    # establish data connection to tracker
    state=True
    acknowledged, timeout = tracker.opengaze._send_message('SET', \
        'ENABLE_SEND_DATA', \
        values=[('STATE', int(state))], \
        wait_for_acknowledgement=True)
    init_t1 = time.time()
    init_t = round(init_t1-init_t0,3)
    msg_init = "Initiation time spent: {} s".format(str(init_t))
    print(msg_init)
    #print(acknowledged)
    
    # set up experiment and perform calibration

    # display instructions
    scr.draw_text(text="Please keep still during the experiment. Click the mouse to start the calibration.", fontsize=20)
    disp.fill(scr)
    disp.show()

    # wait for a keypress
    #kb.get_key(keylist=None, timeout=None, flush=True)
    mouse.get_clicked()


    cal_t0 = time.time()
    # calibrate the eye tracker
    tracker.calibrate()
    cal_t1 = time.time()
    cal_t = round(cal_t1-cal_t0,3)
    msg_cal = "Calibration time spent: {} s".format(str(cal_t))
    print(msg_cal)


    screenshot_name = time.strftime("%y%m%d-%H%M%S", time.localtime())+"_calibration.png"
    disp.make_screenshot(filename=os.path.join(PREDDIR, screenshot_name))
    # perform experiment

    scr.clear()
    txt = f'You will see {ntrials} images in sequence, each containing one or more tables. '
    txt = txt + f'Please gaze at a table of interest in each image for {MAXTRIALTIME//1000} seconds.'
    txt = txt + '\n\n' + f'Click the mouse to start.'
    scr.draw_text(text= txt, fontsize=20)
    disp.fill(scr)
    disp.show()
    
    # wait for a keypress
    #kb.get_key(keylist=None, timeout=None, flush=True)
    mouse.get_clicked()

    exp_t0 = time.time()
    # loop through all trials
    for trialnr in range(ntrials):
        
        # PREPARE TRIAL
        # draw the image
        scr.clear()
        scr.draw_image(os.path.join(IMGDIR,images[trialnr]),scale=RESCALE)
    
        # RUN TRIAL
        # start tracking
        tracker.start_recording()
        tracker.log("TRIALSTART %d" % trialnr)
        tracker.log("IMAGENAME %s" % images[trialnr])
        tracker.status_msg("trial %d/%d" % (trialnr+1, ntrials))
    
        # present image
        disp.fill(scr)
        t0 = disp.show()
        tracker.log("image online at %d" % t0)
    
        start_time = time.time()
        while True:  
            # key is pressed or time exceeds limit
            if keyboard.is_pressed('space'):
                break
            elif (time.time() - start_time) > (MAXTRIALTIME / 1000):  
                break 
    
        timer.pause(FIXATIONTIME)
    
        # reset screen
        disp.fill()
        t1 = disp.show()
        tracker.log("image offline at %d" % t1)

    
        # stop recording
        tracker.log("TRIALEND %d" % trialnr)
        tracker.stop_recording()
    
        # TRIAL AFTERMATH
        # bookkeeping
    
    
        # inter trial interval
        timer.pause(500)
    
    # close experiment
    tracker._elog('pygaze experiment report end')
    # loading message
    scr.clear()
    exp_t1 = time.time()
    exp_t = round(exp_t1-exp_t0,3)
    msg_exp = "Experiment time spent: {} s".format(str(exp_t))
    print(msg_exp)

    #log time
    tracker._elog(msg_init)
    tracker._elog(msg_cal)
    tracker._elog(msg_exp)

    scr.draw_text(text="Transferring the data file, please wait...", fontsize=20)
    disp.fill(scr)
    disp.show()

    close_t0 = time.time()
    # neatly close connection to the tracker
    # (this will close the data file, and copy it to the stimulus PC)
    tracker.close()
    # close the logfile
    #log.close()

    close_t1 = time.time()
    close_t = round(close_t1-close_t0,3)
    msg_close = "Disconnection time spent: {} s".format(str(close_t))
    print(msg_close)

    ############################## preprocess ##################################

    scr.clear()
    scr.draw_text(text="Processing data...", fontsize=20)
    disp.fill(scr)
    disp.show()

    prep_t0 = time.time()
    for file in os.listdir(DATADIR):
        if not file.startswith(LOGFILENAME):
            continue
        elif file.endswith(".tsv"): d = file 
        elif file.endswith(".txt"): l = file
    #print('d:',d, '\nl:',l)
    data = process_data(DATADIR, d, l, ntrials)

    data = data[data.IMGID != -1]
    filename = os.path.join(DATADIR,'trim_edge_'+time.strftime("%y%m%d-%H%M%S", time.localtime())+'.csv')
    trimmed = trim_edge(data=data, filename = filename)

    prep_t1 = time.time()
    prep_t = round(prep_t1-prep_t0,3)
    msg_prep = "Preprocessing time spent: {} s".format(str(prep_t))
    print(msg_prep)

    #plot heapmap and scanpath
    for img in range(len(images)):
        imgname = images[img]
        d = trimmed[trimmed.IMGID == imgname.upper()]

        HPATH = os.path.join(PLOTDIR, imgname.rstrip('.png')+'_heatmap.png')
        SPATH = os.path.join(PLOTDIR, imgname.rstrip('.png')+'_scanpath.png')
        imagefile = os.path.join(IMGDIR,imgname.lower())
        draw_heatmap(d, imagefile=imagefile, durationweight=True, alpha=0.3, savefilename=HPATH)
        #draw_scanpath(d, imagefile=imagefile, alpha=0.3, savefilename=SPATH)

    pred_t0 = time.time()
    m = count_density(data=trimmed, boxes=coords, imglist = images)
    pred_t1 = time.time()
    pred_t = round(pred_t1-pred_t0,3)
    msg_pred = "Prediction time spent: {} s".format(str(pred_t))
    print(msg_pred)


    t = open(os.path.join(PREDDIR, 'density.txt'),"w")
    t.write(str(list(m.items())))
    t.close()
    
    ############################### display results #############################
    scr.clear()
    txt = f'The tables have been saved to the local drive. '+'\n'
    txt = txt + f'Click the mouse to see the potential Area of Interest (AOI).'
    scr.draw_text(text= txt, fontsize=20)
    disp.fill(scr)
    disp.show()

    # wait for a keypress or a mouseclick
    #kb.get_key(keylist=None, timeout=None, flush=True)
    mouse.get_clicked()

    for trialnr in range(ntrials):
    
        # PREPARE TRIAL
        # draw the image
        scr.clear()
        scr.draw_image(os.path.join(PREDDIR,images[trialnr]),scale=1.0)
        disp.fill(scr)
    
    
        img_name = images[trialnr]
        # predicted density
        v = m[img_name]
        i = v.index(max(v))
        _x,_y,_w,_h = coords[coords.img_name == img_name.split('.')[0]].boxes.values[0][i]
        #in pixels
        edge_x = (DISPSIZE[0] - IMGSIZE[0])//2
        edge_y = (DISPSIZE[1] - IMGSIZE[1])//2
        x,y,w,h = (_x-_w/2)*IMGSIZE[0]+edge_x, (_y-_h/2)*IMGSIZE[1]+edge_y, _w*IMGSIZE[0], _h*IMGSIZE[1]
        scr.draw_rect(colour='blue', x=x, y=y, w=w, h=h, pw=5)
        disp.fill(scr)
    
        display_str = "{}: {}%".format('Density of fixations', v[i])
        scr.draw_text(text=display_str, colour='blue', pos=(int(x+w), int(y+h+10)),centre = False, fontsize=20)
        disp.fill(scr)
    
        txt = f'Click the mouse'+'\n'
        txt = txt + f'to show the' + '\n'+ 'next image'
        scr.draw_text(text=txt, pos=(DISPSIZE[0]-5, DISPSIZE[1]//2), centre = False, fontsize=20)
        disp.fill(scr)
        # present image
        disp.fill(scr)
        disp.show()
        SCRSHOT = 'AOI_prediction_'+img_name
        disp.make_screenshot(filename=os.path.join(PLOTDIR, SCRSHOT))
    
        mouse.get_clicked()
    
        # reset screen
        disp.fill()
        disp.show()


    # exit message
    scr.clear()
    scr.draw_text(text="This is the end of this experiment. Thanks for participating!\n\n(click the mouse to exit)", 
                  fontsize=20)
    disp.fill(scr)
    disp.show()

    t_list = [init_t, cal_t, exp_t, close_t, prep_t, pred_t]

    # wait for a keypress
    #kb.get_key(keylist=None, timeout=None, flush=True)
    mouse.get_clicked()

    # close the Display
    disp.close()
        
    return t_list