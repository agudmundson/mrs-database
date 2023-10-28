
__author__  = 'Aaron Gudmundson'
__email__   = 'aarongudmundsonphd@gmail.com'
__date__    = '2020/12/05'
__version__ = '10.0.0'
__status__  = 'beta'

from matplotlib.widgets import RadioButtons, TextBox, Button
from matplotlib.axes import Axes
import matplotlib.patches as mpatches                                                       # Figure Legends
import matplotlib.image as mpimg
import matplotlib.pyplot as plt                                                             # Plotting
import pandas as pd                                                                         # DataFrames
import numpy as np                                                                          # Arrays
import time as t0                                                                           # Determine Run Time
import subprocess
import struct                                                                               # Reading Binary
import copy
import glob                                                                                 # Bash-like File Reading
import sys                                                                                  # Interaction w/System
import os                                                                                   # Interaction w/Operating System
import re

np.set_printoptions(threshold=np.inf, precision=3, linewidth=300, suppress=False)           # Terminal Numpy Settings
np.seterr(divide='ignore', invalid='ignore')                                                # Terminal Numpy Warnings

class ReadFigure:

	def __init__(self, iname):
		self.iname       = sys.argv[1]
		self.image       = mpimg.imread(iname)

		self.left        = 0
		self.right       = self.image.shape[1]
		self.bottom      = self.image.shape[0]
		self.top         = 0

		self.fig_left    = 0
		self.fig_right   = self.image.shape[1]
		self.fig_bottom  = self.image.shape[0]
		self.fig_top     = 0

		self.eventx      = 0
		self.eventy      = 0 
		self.nclick      = 0

		self.xaxis       = np.linspace(self.left  , self.right, self.image.shape[1])
		self.yaxis       = np.linspace(self.bottom, self.top  , self.image.shape[0])


		## Display Image
		self.fig,self.ax = plt.subplots(figsize=(14,6))
		plt.subplots_adjust(left=0.25, bottom=.1)
		self.ax.imshow(self.image)

		## Setting up Different Group
		self.ngroups     = 3
		self.group_names = ['Group 1', 'Group 2', 'Group 3']
		self.group_dict  = {'Group 1': [],
							'Group 2': [],
							'Group 3': []}
		self.curr_group  =  'Group 1'
		self.groups      = RadioButtons(plt.axes([0.18, 0.49, 0.14, 0.12]), self.group_names)

		## Setting up Axes Buttons
		self.label       = 'Off'
		self.rad         = RadioButtons(plt.axes([0.05, 0.26, 0.12, 0.40]), ('Off'   , 
																			 'Click' ,
																			 'Left'  , 
																			 'Right' , 
																			 'Bottom', 
																			 'Top'   ))


		## Setting up Text Input for Axes Locations
		self.n_groups    = TextBox(plt.axes([0.18, 0.62, 0.140, 0.04]), 'Type the Number of Groups:', initial='{}'.format(self.ngroups))
		self.nlabel      = self.n_groups.ax.get_children()[0]
		self.nlabel.set_position([0.5,1.8])
		self.nlabel.set_verticalalignment('top')
		self.nlabel.set_horizontalalignment('center')

		self.text_left   = TextBox(plt.axes([0.90, 0.90, 0.100, 0.10]), ' '     , initial='HELLO')
		self.text_left   = TextBox(plt.axes([0.18, 0.41, 0.140, 0.04]), ' '     , initial='0')
		self.text_right  = TextBox(plt.axes([0.18, 0.36, 0.140, 0.04]), ' '     , initial='{}'.format(self.image.shape[1]))
		self.text_bottom = TextBox(plt.axes([0.18, 0.31, 0.140, 0.04]), ' '     , initial='{}'.format(self.image.shape[0]))
		self.text_top    = TextBox(plt.axes([0.18, 0.26, 0.140, 0.04]), ' '     , initial='0')
		
		self.button      = Button( plt.axes([0.05, 0.20, 0.270, 0.05]), 'Update',)

		self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)

		self.groups.on_clicked(self.get_group)
		self.rad.on_clicked(self.get_label)
		self.button.on_clicked(self.update_axes)

		self.n_groups.on_submit(self.ontext_ngroups)
		self.text_left.on_submit(self.ontext_left)
		self.text_right.on_submit(self.ontext_right)
		self.text_bottom.on_submit(self.ontext_bottom)
		self.text_top.on_submit(self.ontext_top)

		self.cid         = self.fig.canvas.mpl_connect('button_press_event', self.click)
		self.ax.axis('off')
		plt.show()

	def get_group(self, label):
		self.curr_group = label

	def ontext_ngroups(self, text):
		
		for ii in range(self.ngroups):
			curr = self.groups.ax.get_children()[ii]
			curr.set_text(' ')

		self.ngroups = int(text)

		groups       = []
		for ii in range(self.ngroups):
			groups.append('Group {}'.format(ii+1))

		self.groups.ax.set_frame_on(False)
		# self.groups.labels = groups
		self.groups  = RadioButtons(plt.axes([0.18, 0.49, 0.14, 0.12]), groups)

	def ontext_left(self, text):
		self.text_left.set_val(text)
		self.fig_left = float(text)

	def ontext_right(self, text):
		self.text_right.set_val(text)
		self.fig_right = float(text)

	def ontext_bottom(self, text):
		self.text_bottom.set_val(text)
		self.fig_bottom = float(text)

	def ontext_top(self, text):		
		self.text_top.set_val(text)
		self.fig_top = float(text)

	def update_axes(self, event):
		figax = 'Figure: x-axis {:4d}-{:4d}  |  y-axis  {:4d}-{:4d}'.format(self.left, self.right, self.bottom, self.top)
		imgax = 'Image: x-axis {:4.0f}-{:4.0f}  |  y-axis {:4.0f}-{:4.0f}'.format(self.fig_left, self.fig_right, self.fig_bottom, self.fig_top)
		self.ax.set_title('{}\n{}'.format(figax, imgax))

		print('Figure Axes:')
		print('    Figure x-axis: {:4d}-{:4d}'.format(self.left, self.right))
		print('    Figure y-axis: {:4d}-{:4d}'.format(self.bottom, self.top))
		print('Image  Axes:')
		print('    Image  x-axis: {:4.0f}-{:4.0f}'.format(self.fig_left, self.fig_right))
		print('    Image  y-axis: {:4.0f}-{:4.0f}'.format(self.fig_bottom, self.fig_top))		

		self.xaxis       = np.linspace(self.fig_left, self.fig_right , self.right  - self.left)
		self.yaxis       = np.linspace(self.fig_top , self.fig_bottom, self.bottom - self.top )

		print('\n\tDistance/Unit:  X = {:8.4f}  |  Y = {:8.4f}  '.format( np.abs(self.xaxis[1] - self.xaxis[0]), np.abs(self.yaxis[1] - self.yaxis[0])))

	def get_label(self, label):
		self.label = label
		print(self.label)

	def click(self, event):

		if event.ydata == None or event.xdata == None:
			return None

		if self.label in ['Left', 'Right', 'Bottom', 'Top']:
			if self.label == 'Left':
				self.left = int(event.xdata)
			elif self.label == 'Right':
				self.right = int(event.xdata)
			elif self.label == 'Bottom':
				self.bottom = int(event.ydata)
			elif self.label == 'Top':
				self.top = int(event.ydata)
		
		elif self.label == 'Click':
			xpoint = int(event.xdata) - self.left
			ypoint = int(event.ydata) - self.bottom
			if xpoint > 0:
				x_     = 'X: {:5d} {:8.4f}'.format(xpoint, self.xaxis[xpoint])
				y_     = 'Y: {:5d} {:8.4f}'.format(ypoint, self.yaxis[ypoint])
				print('{}:  {}  |  {}          |          {:8.4f},{:8.4f}'.format(self.curr_group, x_, y_, self.xaxis[xpoint], self.yaxis[ypoint]))

		
if __name__ == '__main__':

	iname   = sys.argv[1]
	RF      = ReadFigure(iname)