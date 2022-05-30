# the ip address to connect with the eye tracker
ip = '172.20.10.3'

# how long the experimenter looks at the image after pressing 'space' (ms) -- transitioning time
FIXATIONTIME = 1500
# this is the type of eye tracker we will be using
TRACKERTYPE = 'opengaze'

# DISPLAY
# for the DISPTYPE, you can choose between 'pygame' and 'psychopy'; go for
# 'psychopy' if you need millisecond accurate display refresh timing, and go
# for 'pygame' if you experience trouble using PsychoPy
DISPTYPE = 'psychopy'
# the DISPSIZE is the monitor resolution, e.g. (1024,768)
#DISPSIZE = (1920,1080) #(2880, 1800) #(1024,768)
DISPSIZE = (1024,1024)
# image size 
IMGSIZE= (1200,1553)

#RESCALE = 1.1
RESCALE = min(round(DISPSIZE[0] / IMGSIZE[0], 1), round(DISPSIZE[1] / IMGSIZE[1], 1))
# the SCREENSIZE is the physical screen size in centimeters, e.g. (39.9,29.9)
SCREENSIZE = (34.4, 19.3)

# set FULLSCREEN to True for fullscreen displaying, or to False for a windowed
# display, for experiments, set this to True, and for testing, set to False
FULLSCREEN = False
# BGC is for BackGroundColour, FGC for ForeGroundColour; both are RGB guns,
# which contain three values between 0 and 255, representing the intensity of
# Red, Green, and Blue respectively, e.g. (0,0,0) for black, (255,255,255) for
# white, or (255,0,0) for the brightest red
BGC = (0,0,0)
#BGC = (255,255,255)
FGC = (255,255,255)

TEXTSIZE = 12

#Experiment variables
#    The distance between the participant and the display (cm)
SCREENDIST = 60.0                               
#    Maximum length of time for each image to be displayed (ms)
MAXTRIALTIME = 10000         
#    Maximum length of time for fixation (s)
MAXFIXTIME = 5 