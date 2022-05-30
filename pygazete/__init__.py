# -*- coding: utf-8 -*-

# Pygazeplot is a simplified version of the Python module PyGazeAnalyser 
# v-0.1.0 Copyright (C) 2014 created by Edwin S. Dalmaijer for easily 
# analysing eye-tracking data. 
#
#   Pygazeplot is modified by Chloe Liu for creating visualizations of 
#   eye-tracking log data generated from PyGaze experiments.
# 
author = __author__ = "Chloe Liu"
version = u'0.2.1'
#
#
# INFO ABOUT PyGazeAnalyser:
#
# This file is part of PyGaze - the open-source toolbox for eye tracking
#
#	PyGazeAnalyser is a Python module for easily analysing eye-tracking data
#	Copyright (C) 2014  Edwin S. Dalmaijer
#
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>

from pygazete.constants import *
from pygazete.preprocess import *
from pygazete.datasets import *
from pygazete.table_structure import *
from pygazete.gazeplotter import *
from pygazete.table_extraction import *
from pygazete.ocr import *
#from pygazete.evaluate import *
#from pygazete.extraction_baseline import *

acknowledgement = "PygazeTE imported.\n"
acknowledgement += "Some important defaults: \n"
acknowledgement += "\tDISPSIZE = %s \t(Desired size of display in pixels),\n\tSCREENSIZE = %s \t(Size of screen in centimeters),\n\tip = %s \t\t(IP address for Gazepoint connection),\n\t... ...\n" % (str(DISPSIZE), str(SCREENSIZE), ip)
acknowledgement += "Please beware of the default constants and change variables to adapt.\n"
acknowledgement += "You can either reassign the variables or change the default values in the root script from pygazete.constants"
print(acknowledgement)


