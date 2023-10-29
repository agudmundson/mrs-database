
__author__  = 'Aaron Gudmundson'
__email__   = 'aarongudmundsonphd@gmail.com'
__date__    = '2020/12/05'
__version__ = '10.0.0'
__status__  = 'beta'

from matplotlib.widgets import RadioButtons, TextBox, Button 								# Matplotlib Widgets
import matplotlib.patches as mpatches                                                       # Figure Legends
import matplotlib.image as mpimg   															# Image Display
import matplotlib.pyplot as plt                                                             # Plotting
import pandas as pd                                                                         # DataFrames
import numpy as np                                                                          # Arrays
import datetime  																			# Date/Time
import copy 																				# Copy Objects
import sys                                                                                  # Interaction w/System
import os                                                                                   # Interaction w/Operating System

np.set_printoptions(threshold=np.inf, precision=3, linewidth=300, suppress=False)           # Terminal Numpy Settings
np.seterr(divide='ignore', invalid='ignore')                                                # Terminal Numpy Warnings

class ReadFigure:

	def __init__(self, iname):
		self.iname       = sys.argv[1] 														# Input Image Name
		self.image       = mpimg.imread(iname) 												# Read the Image

		self.left        = 0 																# Instantiate Image Axes
		self.right       = self.image.shape[1]  											# 
		self.bottom      = self.image.shape[0]												# 
		self.top         = 0 																# 

		self.fig_left    = 0 																# Instantiate Figure Axes
		self.fig_right   = self.image.shape[1] 												# 
		self.fig_bottom  = self.image.shape[0] 												# 
		self.fig_top     = 0 																# 

		self.eventx      = 0 																# x-coordinate
		self.eventy      = 0  																# y-coordinate

		self.xaxis       = np.linspace(self.left  , self.right, self.image.shape[1]) 		# Define x-axis
		self.yaxis       = np.linspace(self.bottom, self.top  , self.image.shape[0]) 		# Define y-axis


		## Display Image
		self.fig,self.ax = plt.subplots(figsize=(14,6)) 									# Instantiate Figure
		plt.subplots_adjust(left=0.25, bottom=.1) 											# Move Figure to Right
		self.ax.imshow(self.image) 															# Display Image within Figure


		## Setting up Different Group
		self.ngroups     = 3 																# Default to 3 Groups
		self.group_names = ['Group 1', 'Group 2', 'Group 3'] 								# Default Group Names
		self.values_dict = {'Time'   : [], 													# Default Values Dictionary
							'N_Group': [], 													# 
						    'Group'  : [], 													# 
							'img-X'  : [], 													# 
							'img-Y'  : [], 													# 
							'fig-X'  : [], 													# 
							'fig-Y'  : []} 													# 

		self.curr_group  =  'Group 1' 														# Set Initial Group
		self.groups      = RadioButtons(plt.axes([0.18, 0.49, 0.14, 0.12]),self.group_names)# Groups Radio Selection

		## Setting up Axes Buttons
		self.label       = 'Off' 															# Set Initial Options Selection
		self.rad         = RadioButtons(plt.axes([0.05, 0.26, 0.12, 0.40]), ('Off'   ,  	# Options Radio Selection
																			 'Click' ,
																			 'Left'  , 
																			 'Right' , 
																			 'Bottom', 
																			 'Top'   ))


		## Setting up Text Input for Axes Locations
		self.n_groups    = TextBox(plt.axes([0.18, 0.62, 0.140, 0.04]), 'Type the Number of Groups:', initial='{}'.format(self.ngroups))
		
		self.nlabel      = self.n_groups.ax.get_children()[0] 								# Get Group Text Label
		self.nlabel.set_position([0.5,1.8]) 												# Adjust Position Above
		self.nlabel.set_verticalalignment('top') 											# Center
		self.nlabel.set_horizontalalignment('center') 										# Top


		## Text Boxes for Figure Axes
		self.text_left   = TextBox(plt.axes([0.18, 0.41, 0.140, 0.04]), ' '     , initial='0')
		self.text_right  = TextBox(plt.axes([0.18, 0.36, 0.140, 0.04]), ' '     , initial='{}'.format(self.image.shape[1]))
		self.text_bottom = TextBox(plt.axes([0.18, 0.31, 0.140, 0.04]), ' '     , initial='{}'.format(self.image.shape[0]))
		self.text_top    = TextBox(plt.axes([0.18, 0.26, 0.140, 0.04]), ' '     , initial='0')
		
		self.button      = Button( plt.axes([0.05, 0.20, 0.270, 0.05]), 'Update',) 			# Update Button

		self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id) 		# Key Press Handler

		## Buttons
		self.button.on_clicked(self.update_axes) 											# Button Clicks
		self.groups.on_clicked(self.set_group) 												
		
		## Radio Selections
		self.n_groups.on_submit(self.ontext_ngroups)
		self.rad.on_clicked(self.set_label)

		## Text Boxes
		self.text_left.on_submit(self.ontext_left) 											# Left Axis Click Point
		self.text_right.on_submit(self.ontext_right)										# Right Axis Click Point
		self.text_bottom.on_submit(self.ontext_bottom)										# Bottom Axis Click Point
		self.text_top.on_submit(self.ontext_top)											# Top Axis Click Point

		self.fig.canvas.mpl_connect('close_event', self.on_close) 							# Save .csv upon close
		self.cid         = self.fig.canvas.mpl_connect('button_press_event', self.click) 	# Click Events
		self.ax.axis('off') 																# Remove Outer Axes
		plt.show() 																			# Display

	def set_group(self, label): 															# Set Current Group
		self.curr_group = label 															# Set Group Label
		print('Selected: ', self.curr_group) 												# Output User Selection

	def ontext_ngroups(self, text):
		
		self.groups.ax.set_frame_on(False) 													# Remove Group Radio Select Box
		for ii in range(self.ngroups): 														# Remove Each Group from Radio Select Box
			curr = self.groups.ax.get_children()[ii] 										# Current 
			curr.set_text(' ') 																# Set Text Blank

		self.ngroups     = int(text) 														# Update Number of Groups

		self.values_dict['Time'   ].append(datetime.datetime.now()) 		 				# Update Values - Time
		self.values_dict['N_Group'].append('Update') 										# Update Values - Number of Groups
		self.values_dict['Group'  ].append(' ') 					 						# Update Values - Group
		self.values_dict['img-X'  ].append(' ') 											# Update Values - Image - X
		self.values_dict['img-Y'  ].append(' ') 											# Update Values - Image - Y
		self.values_dict['fig-X'  ].append(' ') 											# Update Values - Figure X
		self.values_dict['fig-Y'  ].append(' ') 											# Update Values - Figure Y

		groups       = [] 																	# Input for New Group Radio Select
		for ii in range(self.ngroups): 														# Iterate Through User Input Number
			groups.append('Group {}'.format(ii+1)) 											# Create Group Name

		self.groups  = RadioButtons(plt.axes([0.18, 0.49, 0.14, 0.12]), groups) 			# Instantiate New Group Radio Select

	def ontext_left(self, text): 															# Modify Left Axis Text box
		self.text_left.set_val(text) 														# Set Text Value
		self.fig_left = float(text) 														# Set Figure Left Axis Start

	def ontext_right(self, text):															# Modify Right Axis Text box
		self.text_right.set_val(text) 														# Set Text Value
		self.fig_right = float(text)														# Set Figure Right Axis Start

	def ontext_bottom(self, text):															# Modify Bottom Axis Text box
		self.text_bottom.set_val(text) 														# Set Text Value
		self.fig_bottom = float(text)														# Set Figure Bottom Axis Start

	def ontext_top(self, text):																# Modify Top Axis Text box
		self.text_top.set_val(text) 														# Set Text Value
		self.fig_top = float(text)															# Set Figure Top Axis Start

	def update_axes(self, event):
		figax = 'Figure: x-axis {:4d}-{:4d}  |  y-axis  {:4d}-{:4d}'.format(self.left, self.right, self.bottom, self.top)
		imgax = 'Image: x-axis {:4.0f}-{:4.0f}  |  y-axis {:4.0f}-{:4.0f}'.format(self.fig_left, self.fig_right, self.fig_bottom, self.fig_top)
		self.ax.set_title('{}\n{}'.format(figax, imgax))

		print('Figure Axes:') 																# Output New Figure and Image Axes
		print('    Figure x-axis: {:4d}-{:4d}'.format(self.left, self.right)) 				# Figure x-axis
		print('    Figure y-axis: {:4d}-{:4d}'.format(self.bottom, self.top)) 				# Figure y-axis
		print('Image  Axes:')
		print('    Image  x-axis: {:4.0f}-{:4.0f}'.format(self.fig_left, self.fig_right))	# Image x-axis
		print('    Image  y-axis: {:4.0f}-{:4.0f}'.format(self.fig_bottom, self.fig_top))	# Image y-axis		

		self.xaxis = np.linspace(self.fig_left, self.fig_right , self.right  - self.left) 	# Reset the linear-spaced x-axis
		self.yaxis = np.linspace(self.fig_top , self.fig_bottom, self.bottom - self.top )   # Reset the linear-spaced y-axis

		print('\n\tDistance/Unit:  X = {:8.4f}  |  Y = {:8.4f}  '.format( np.abs(self.xaxis[1] - self.xaxis[0]), np.abs(self.yaxis[1] - self.yaxis[0])))

	def set_label(self, label): 															# Set Options Label
		self.label = label 																	# Set Options Label
		print('Selected: ', self.label) 													# Output Option Selected

	def click(self, event): 																# Click Events		

		if event.ydata == None or event.xdata == None: 										# User Clicked Outside of Figure
			return None  																	# Do Nothin

		if self.label in ['Left', 'Right', 'Bottom', 'Top']: 								# Modifying Axes
			if self.label == 'Left': 														# Modify Left
				self.left = int(event.xdata)
			elif self.label == 'Right': 													# Modify Right
				self.right = int(event.xdata)
			elif self.label == 'Bottom': 													# Modify Bottom
				self.bottom = int(event.ydata)
			elif self.label == 'Top':														# Modify Top
				self.top = int(event.ydata)
		
		elif self.label == 'Click': 														# User Clicks
			xpoint = int(event.xdata) - self.left 											# Set x-point
			ypoint = int(event.ydata) - self.bottom 										# Set y-point
			
			if xpoint > 0: 																	# Click was within range
				x_     = 'X: {:5d} {:8.4f}'.format(xpoint, self.xaxis[xpoint]) 				# output x-string 
				y_     = 'Y: {:5d} {:8.4f}'.format(ypoint, self.yaxis[ypoint]) 				# output y-string 
				print('{}:  {}  |  {}          |          {:8.4f},{:8.4f}'.format(self.curr_group, x_, y_, self.xaxis[xpoint], self.yaxis[ypoint]))

				self.values_dict['Time'   ].append(datetime.datetime.now()) 				# Update Values - Time
				self.values_dict['N_Group'].append(self.ngroups   )							# Update Values - Number of Groups
				self.values_dict['Group'  ].append(self.curr_group) 						# Update Values - Group
				self.values_dict['img-X'  ].append(xpoint) 									# Update Values - Image - X
				self.values_dict['img-Y'  ].append(ypoint) 									# Update Values - Image - Y
				self.values_dict['fig-X'  ].append(self.xaxis[xpoint])						# Update Values - Figure X
				self.values_dict['fig-Y'  ].append(self.yaxis[ypoint])						# Update Values - Figure Y

	def on_close(self, event): 																# Close Event
		df   = pd.DataFrame(self.values_dict) 												# Create DataFrame from Values Dictionary
		print('\n') 																		# Leave Some Space
		print(df) 																			# Print the DataFrame
		print('\n') 																		# Leave Some Space

		ext  = len(self.iname.split('.')[-1]) + 1 											# Length of Image Extension
		df.to_csv('{}.csv'.format(self.iname[:-ext])) 										# Replace Image Extension with .csv

if __name__ == '__main__': 																	# Script is Called

	iname   = sys.argv[1] 																	# Image Name
	RF      = ReadFigure(iname) 															# Create Class Object - Start Matplotlib Application