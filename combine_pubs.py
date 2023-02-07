__author__  = 'Aaron Gudmundson'
__email__   = 'agudmund@uci.edu'
__date__    = '2021/09/01'
__version__ = '1.0.0'
__status__  = 'beta'


import matplotlib.pyplot as plt 															# Plotting
import scipy.io as sio 																		# SciPy
import pandas as pd 																		# DataFrames
import numpy as np 																			# Arrays
import unicodedata
import time as t0																		    # Determine Run Time
import string 																				# String Operations 
import struct 																				# Reading Binary
import glob 																				# Bash-like File Reading
import copy
import ast
import sys 																					# Interaction w/System
import os 																					# Interaction w/Operating System
import re

np.set_printoptions(threshold=np.inf, precision=3, linewidth=300, suppress=True) 			# Terminal Numpy Settings
np.seterr(divide='ignore', invalid='ignore') 												# Terminal Numpy Warnings

class CombinePublications:
	
	def __init__(self, basedir):	
		
		self.basedir  = basedir 															# Base Directory

		self.bib_req  = ['title'        , 'author' , 'journal'  , 'year'     , 'volume'    ]# bibtex Required Fields
		self.bib_opt  = ['address'      , 'annote' , 'booktitle', 'email'    , 'chapter'   ,# bibtex Optional Fields
					     'crossref'     , 'edition', 'editor'   , 'series'   , 'key'       ,
					     'organization' , 'note'   , 'number'   , 'publisher', 'school'    ,
					     'howpublished' , 'type'   , 'month'    , 'issn'     , 'pages'     , 
					     'institution'  , 'url'    , 'language' , 'keywords' , 'abstract'  , 
					     'document_type', 'doi'    , 'keywords' , 'source'   , 'conference',
					     'art_number'   , 'isbn'   , 'citation' ,                          ]
		self.bib_cst  = ['editorial'    ,                                   			   ] # bibtex Custom Fields
		
		self.bib_flds = copy.deepcopy(self.bib_req) 										# bibtxt All Fields
		self.bib_flds.extend(copy.deepcopy(self.bib_opt))  									# bibtxt All Fields
		self.bib_flds.extend(copy.deepcopy(self.bib_cst))  									# bibtxt All Fields

		# self.bib_cite = ['Year', 'Author', 'Volume', 'Journal'] 							# bibtxt's 'CiteKey' will be the 'Key' Entry in pub_dict (first_author)
		self.bib_cite = ['Year', 'Author', 'Volume'] 										# bibtxt's 'CiteKey' will be the 'Key' Entry in pub_dict (first_author)
		# self.bib_cite = ['Year', 'Author'] 													# bibtxt's 'CiteKey' will be the 'Key' Entry in pub_dict (first_author)
		self.pub_dict = {} 																	# Empty Dictionary for Publications

		self.warnings = '' 																	# Logging - Note
		self.errors   = '' 																	# Logging - Catch
		self.duplicate= '' 																	# Logging - Acknowledge

		self.total    = 0 																	# Publication - Total
		self.abstract = 0 																	# Publication - Has Abstract
		self.book     = 0  																	# Publication - is Book
		self.conf     = 0 																	# Publication - from Conference
		self.no_auth  = 0 																	# Publication - has NO Author
		self.non_eng  = 0																	# Publication - NOT in English
		self.unk_lng  = 0																	# Publication - Language NOT Listed
		self.non_art  = 0 																	# Publication - Not Article (Misc.)
		self.review   = 0 																	# Publication - Review

		self.path0n   = 0
		self.path1n   = 0
		self.path2n   = 0

	def read_articles(self,fnames):
		self.fnames = {} 																	# Names of Filenames and Number of Articles

		csv  = [] 																			# Citations - In CSV
		bib  = [] 																			# Citations - In bibtxt
		ris  = [] 																			# Citations - In ris 

		for ii in range(len(fnames)):
			self.fnames[fnames[ii]] = 0 													# Number of Articles from Each
			
			if fnames[ii][-3:].lower() == 'csv':
				csv.append(fnames[ii])
			elif fnames[ii][-3:] == 'bib':
				bib.append(fnames[ii])
			elif fnames[ii][-3:] == 'ris':
				ris.append(fnames[ii])

		self.read_bibtxt(bib)
		self.read_csv(csv)

		print('\n    Breakdown: ')
		for ii in range(len(fnames)):
			print('\t{:5d} Articles | {}'.format(self.fnames[fnames[ii]], fnames[ii]))
		
		self.total = len(list(self.pub_dict.keys()))
		print('\n\t{:5d} Articles |  Total ({:5d} Have Abstracts )\n'.format(self.total, self.abstract))

	def read_bibtxt(self,fnames): 															# Read bibtxt Files (*.bib)

		for ii in range(len(fnames)):
			pub       = open('{}/{}'.format(self.basedir,fnames[ii]), 'r', errors='ignore') # Open Publication
			ptext     = pub.read() 															# Read in as String
			pub.close() 																	# Close File

			ptext     = ptext.lower()
			ptext     = ptext.split('@article')[1:] 										# Split By Individual Entry	 ## print('\n\n', ptext, '\n\n')
			total     = len(ptext) 															# Total Number of Articles

			prog      = 0
			prog_sym  = '. ' 																# Symbol for Progress Bar
			tstart    = t0.time()
			for jj in range(len(ptext)):
				if jj > total/100 * prog:
					prog +=1
					if prog != 0 and prog % 10 == 0:
						print('{:<20}'.format(prog_sym * (prog//10)), '{:3d}% |'.format(prog), end='\r' ) # Percent Bar

				pind  = ptext[jj].split('\n') 												# Split Each Line
				dind  = {'title'  : [], 													# bibtxt Required Fields
						 'author' : [], 													# bibtxt Required Fields
						 'journal': [], 													# bibtxt Required Fields
						 'year'   : [], 													# bibtxt Required Fields
						 'volume' : [], 													# bibtxt Required Fields 	
						 'doi'    : []} 													# Project Required

				abstr = 0
				for kk in range(len(pind)):
					if '={' in pind[kk]:

						if len(pind[kk].split('={')) > 2:
							self.errors = '{}{:2d} {:5d} {:2d} Multiple Entries Detected ***\n'.format(self.errors, ii, jj, kk)

						pkey = pind[kk].split('={')[0].strip()
						if pkey == 'abstract':
							abstr = 1
						else:
							abstr = 0
						pval = pind[kk].split('={')[1].strip()
						pval = pval.replace('{', '')
						pval = pval.replace('}', '')
						pval = pval.rstrip(',')
						pval = pval.rstrip('\n')

						if pkey not in ['researcherid-numbers', 'orcid-numbers', 'unique-id', 'firstauthor', 'article-number', 'pmcid', 'citation']:
							dind[pkey] = pval

						if pkey not in self.bib_flds:
							self.warnings = '{}{:2d} {:5d} {:2d} Unidentified Key: {}'.format(self.warnings, ii, jj, kk, pkey)						
					
					elif abstr == 1:
						pval = pind[kk].strip()						
						pval = pval.replace('{', '')
						pval = pval.replace('}', '')
						pval = pval.rstrip(',')
						pval = pval.rstrip('\n')

						dind[pkey] = '{}{}'.format(dind[pkey], pval)

				self.populate_dict(dind)

			self.fnames[fnames[ii]] = total

			txt = '{:<20} 100% |'.format(prog_sym * (prog//10))								# Progress Bar
			txt = '{} {:5d} Articles in {:5.2f} sec'.format(txt, total, t0.time()-tstart) 	# N Articles and Time to Upload
			print('{} {}'.format(txt, fnames[ii]))

	def read_csv(self,fnames):
		for ii in range(len(fnames)):
			df   = pd.read_csv('{}/{}'.format(self.basedir, fnames[ii]), engine='python') 	# Read in CSV format Citations
			cols = list(df.columns) 														# Columns in CSV

			total     = df.shape[0] 														# Total Number of Articles
			prog      = 0
			prog_sym  = '. ' 																# Symbol for Progress Bar
			tstart    = t0.time()
			for jj in range(df.shape[0]):
				if jj > total/100 * prog:
					prog +=1
					if prog != 0 and prog % 10 == 0:
						print('{:<20}'.format(prog_sym * (prog//10)), '{:3d}% |'.format(prog), end='\r' ) # Percent Bar

				dind  = {'title'  : [], 													# bibtxt Required Fields
						 'author' : [], 													# bibtxt Required Fields
						 'journal': [], 													# bibtxt Required Fields
						 'year'   : [], 													# bibtxt Required Fields
						 'volume' : [],} 													# bibtxt Required Fields

				for kk in range(len(cols)):
					value = df[cols[kk]][jj]

					if cols[kk]  == 'author':
						value     = unicodedata.normalize('NFD', value).encode('ascii', 'ignore')
						value     = value.decode()
						value     = value.replace(';', ' and ')
						value     = value.replace(',', ' and ')

					if cols[kk]  == 'issn':	
						if 'elecissn' in df.columns:
							if df.elecissn[jj] is not np.nan:
								eissn = str(df.elecissn[jj]).strip() 							# ISSN - Electronic
							else:
								eissn = ''
						
						if df.issn[jj] is not np.nan:
							pissn = str(df.issn[jj]).strip() 								# ISSN - Print
						else:
							pissn = ''
						
						if df.isbn[jj] is not np.nan:
							isbn  = str(df.isbn[jj]).strip()
							value = '["{}","{}","{}"]'.format(isbn, eissn, pissn) 			# Include ISBN
						else:
							value = '["{}","{}"]'.format(eissn, pissn) 						# Just elec and print ISSN

					if isinstance(value, float) and value is not np.nan:
						try:
							value     = int(value) 
						except:
							print(cols[kk], end='\r')
					if value is np.nan:
						dind[cols[kk]] = []
					
					elif value is not np.nan:
						try:
							value          = str(value)
							dind[cols[kk]] = value
						except Exception as e:
							self.warnings  = '{}{:2d} {:5d} {:2d} NOT a String: {}'.format(self.warnings, ii, jj, kk, cols[kk], df.title[jj])						
				
				self.populate_dict(dind) 													# Add to Dict, Check Duplicates, Abstract, etc.
			self.fnames[fnames[ii]] = total

			txt = '{:<20} 100% |'.format(prog_sym * (prog//10))								# Progress Bar
			txt = '{} {:5d} Articles in {:5.2f} sec'.format(txt, total, t0.time()-tstart) 	# N Articles and Time to Upload
			print('{} {}'.format(txt, fnames[ii]))

	def populate_dict(self, dind):
		
		author1 = str(dind['author']).split(' and ')[0] 									# First Author
		author1 = unicodedata.normalize('NFD', author1).encode('ascii', 'ignore')
		author1 = author1.decode()
		if ',' in author1: 																	# First Author (Last Name, First, Mid-Initial)
			author1 = author1.split(',')[0] 												# First Author Last Name
			author1 = ''.join(author1)														# First Author
			author1 = author1.replace(' ', '') 												# First Author Remove Spaces
		else:
			author1 = author1.split(' ')[0] 												# First Author Last Name Only
			author1 = author1.rstrip(',') 													# First Author Ensure no commas 

		journal = str(dind['journal']).lstrip().rstrip() 									# Full Journal Name w/o spaces
		journal = journal.split(':')[0] 													# Full Journal Remove Extra Qualifiers
		journal = journal.replace(' ', '_') 												# Full Journal Name w/o spaces

		if isinstance(dind['volume'], list):
			try:
				vol = dind['citation'].split(';')[1]
				vol = vol.split('(')[0]
				vol = vol.split(':')[0]
				dind['volume'] = copy.deepcopy(vol)
			except Exception as e:
				pass

		if len(self.bib_cite) == 3:
			ckey    = '{}_{}_{}'.format(dind['year'], author1, dind['volume']) 				# bibtxt's 'Citation Key' and self.pub_dict 'Key'
		elif len(self.bib_cite) == 2:
			ckey    = '{}_{}'.format(dind['year'], author1) 								# bibtxt's 'Citation Key' and self.pub_dict 'Key'

		ckey    = ckey.lower()

		if ckey in list(self.pub_dict.keys()): 												# Publication Already Present
			self.path0n   +=1
			self.duplicate = '{}{}\n'.format(self.duplicate, ckey) 							# Note the Duplicate Publication

			if 'abstract' in list(dind.keys()):	 											# Determine if this entry has Abstract
				self.path1n   +=1
				self.pub_dict[ckey] = dind 													# Use Entry with Abstract
				self.abstract +=1

		else:
			if author1 != 'None':
				self.path2n   +=1
				self.pub_dict[ckey] = dind 													# Use Entry with Abstract

				self.total +=1 																# Note Number of Publications Included
				if 'abstract' in list(dind.keys()):
					self.abstract +=1

	def collapse_duplicates(self, display=False):
		dupl   = self.duplicate.split('\n')
		dupl   = sorted(list(set(dupl)))
		dupl.remove('')

		txt    = 'Duplicates found in Initial Upload'
		print('\t{:5d} {:<40} (Removed)'.format(len(dupl), txt))
		if display == True:
			for ii in range(len(dupl)):
				print('{:5d} {}: {}'.format(ii, dupl[ii], self.pub_dict[dupl[ii]]['title']))

	def find_editorials(self, remove=False):
		pkeys        = sorted(list(self.pub_dict.keys()))

		for ii in range(len(pkeys)):
			if 'editorial' in list(self.pub_dict[pkeys[ii]].keys()):
				if bool(self.pub_dict[pkeys[ii]]['editorial']) == True:
					self.non_art +=1
					
					if remove == True:
						del self.pub_dict[pkeys[ii]]

		txt = 'Not an Article (Other Misc. Print)'
		if remove == False:
			print('\t{:5d} {:<40} (Did NOT Remove)'.format(self.non_art, txt))
		else:
			print('\t{:5d} {:<40} (Removed)'.format(self.non_art, txt))

	def find_conferences(self, remove=False):
		pkeys        = sorted(list(self.pub_dict.keys()))

		for ii in range(len(pkeys)):
			pub      = self.pub_dict[pkeys[ii]]['journal'].lower()
			if 'conference' in pub or 'proceeding' in pub:
				self.conf +=1
				
				if remove == True:
					del self.pub_dict[pkeys[ii]]

			elif 'conference' in list(self.pub_dict[pkeys[ii]].keys()):
				if isinstance(self.pub_dict[pkeys[ii]]['conference'], list):
					self.pub_dict[pkeys[ii]]['conference'] = 'false'

				elif self.pub_dict[pkeys[ii]]['conference'].lower() == 'true':
					self.conf +=1
					
					if remove == True:
						del self.pub_dict[pkeys[ii]]

		txt = 'Conference Posters/Presentations'
		if remove == False:
			print('\t{:5d} {:<40} (Did NOT Remove)'.format(self.conf, txt))
		else:
			print('\t{:5d} {:<40} (Removed)'.format(self.conf, txt))

	def find_no_authors(self, remove=False):
		pkeys        = sorted(list(self.pub_dict.keys()))

		for ii in range(len(pkeys)):
			if self.pub_dict[pkeys[ii]]['author'] == []:
				self.no_auth +=1
				
				if remove == True:
					del self.pub_dict[pkeys[ii]]

		txt = 'Articles without an Author'
		if remove == False:
			print('\t{:5d} {:<40} (Did NOT Remove)'.format(self.no_auth, txt))
		else:
			print('\t{:5d} {:<40} (Removed)'.format(self.no_auth, txt))

	def find_non_english(self, remove=False):
		pkeys        = sorted(list(self.pub_dict.keys()))

		for ii in range(len(pkeys)):
				try:
					if 'eng' not in self.pub_dict[pkeys[ii]]['language']: 					# Language Other Than English
						self.non_eng +=1 													# Count
						
						if remove == True:													# Remove Non-English
							del self.pub_dict[pkeys[ii]]

				except Exception as e: 														# Language Not Listed
						self.unk_lng +=1

		txt0 = 'Non-English Articles'
		txt1 = 'Unlisted Language Articles'
		if remove == False:
			print('\t{:5d} {:<40} (Did NOT Remove)'.format(self.non_eng, txt0))
			print('\t{:5d} {:<40} (Did NOT Remove)'.format(self.unk_lng, txt1))
		else:
			print('\t{:5d} {:<40} (Removed)'.format(self.non_eng, txt0))
			print('\t{:5d} {:<40} (Did NOT Remove)'.format(self.unk_lng, txt1))

	def find_books(self,remove=False):
		pkeys        = sorted(list(self.pub_dict.keys()))

		for ii in range(len(pkeys)):
			pub = self.pub_dict[pkeys[ii]]

			if 'issn' in list(pub.keys()): 										# Determine if issn is included
				issn = pub['issn']
				
				if issn[0] == '[' or issn[-1] == ']': 							# Determine if this is List
					try:

						issn = ast.literal_eval(issn) 							# Try Converting to List
						loc  = 0
						for jj in range(len(issn)):
							if loc == 0: 										# Have NOT Identfied Book Yet
								if '978-' in issn[jj] or '979-' in issn[jj]: 	# Determine if ISBN
									issn = issn[jj] 							# Use this ISSN
									loc += 1 									# Break Loop
						if loc == 0: 											# Couldn't find an ISBN
							issn = issn[0]										# Use 1st String

					except Exception as e: 										# Was NOT List, but has Brackets
						self.warnings = '{}{:2d} {:5d} Failed ISSN List Conversion*** | {}\n'.format(self.warnings, ii, jj, e)
			else: 																	# 
				issn = '0'

			issn = issn.replace('-', '')
			issn = issn.split(' ')[0]
			
			if len(issn) == 10 or len(issn) == 13: 								# Identified ISBN instead of ISSN
				del self.pub_dict[pkeys[ii]] 									# Delete Entry
				self.book  += 1 												# Note Number of Books
			elif len(issn) == 0:
				self.warnings = '{}{:2d} {:5d} No ISSN Found *** {:3d} {}\n'.format(self.warnings, ii, jj, len(issn), pkeys[ii])
			elif issn != '0' and len(issn) != 8:
				self.errors = '{}{:2d} {:5d} Invalid ISSN *** {:3d} {}\n'.format(self.errors, ii, jj, len(issn), issn)

		txt = 'Identified as Books'
		if remove == False:
			print('\t{:5d} {:<40} (Did NOT Remove)'.format(self.book, txt))
		else:
			print('\t{:5d} {:<40} (Removed)'.format(self.book, txt))

	def find_exclusions(self, remove=False):
		pkeys      = sorted(list(self.pub_dict.keys()))

		excl_words = [
					  # 'cest'    , 'chemical exchange saturation',
					  # 'z-spectral', 'z spectral', 'magnetization transfer',
					  # 'breast cancer',
					  # 'biopsy' ,
					  # 'nuclear overhauser effect', 'noe',
					  # 'magic angle spinning',
					  # 'positron emission tomography', 'ligand',
					  # 'urinalysis', 'perchloric acid',
					  # '400 MHz', '400MHz', '500 MHz', '500MHz'
					  # '600 MHz', '600MHz', '800 MHz', '800MHz'
					  # '31p'    , 'phosphorous', 'phosphates', '(31) P',
					  # 'p-31'   , 'c-13'   , 'n-15' , 'Na-23', '(31)P',
					  # '13c'    , 'carbon' , 'poce' , '15n', '23Na'
					  # 'heteronuclear',  'HMQC', 'HSQC',
					  # 'rat'    , 'mouse'  , 'mice' , 'rodent', 
					  # 'dog'    , 'pig'    , 'ex-vivo', 'canine'
					  # 'animal' , 'monkey' , 'transgenic', 'serum', 
					  # 'csf'    , 'cerebral spinal fluid', 
					  # 'liver'  , 'prostate', 'heart', 'kidney', 'muscle',
					  # 'spinal cord', 'Gadolinium', 'Gd3+', 'DTPA',
					  # 'diffusion weighted imaging', 'dwi',
					  # 'diffusion tensor imaging', 'dti',
					  # 'review' , 'meta-analysis', 'conference',
					  # 'methods',
					 ]

		self.dexcl = {}
		for ii in range(len(excl_words)):
			self.dexcl[excl_words[ii]] = []

		for ii in range(len(pkeys)):		
			pub = self.pub_dict[pkeys[ii]]

			txt = pub['title'] 
			if 'abstract' in list(pub.keys()):
				txt = '{}\n{}'.format(txt, pub['abstract'])
			if 'keywords' in list(pub.keys()):
				txt = '{}\n{}'.format(txt, pub['keywords'])

			txt = txt.lower()

			for jj in range(len(excl_words)):
				if excl_words[jj] in txt:
					self.dexcl[excl_words[jj]].append(pkeys[ii])

		for ii in range(len(excl_words)):
			clist = self.dexcl[excl_words[ii]]
			print('\n{}'.format(excl_words[ii]), len(clist))
			for jj in range(len(clist)):
				print('\t{:<30} '.format(clist[jj]), self.pub_dict[clist[jj]]['title'][:140])

	def running_total(self,):

		self.total = len(list(self.pub_dict.keys()))

		pkeys      = sorted(list(self.pub_dict.keys()))
		abs_cnt    = 0

		for ii in range(len(pkeys)):
			pub = self.pub_dict[pkeys[ii]]

			if 'abstract' in list(pub.keys()): 										# Determine if issn is included
				abs_cnt +=1
				
		print('\n\t{:5d} Articles |  Total ({:5d} Have Abstracts )\n'.format(self.total, abs_cnt))

	def find_incomplete(self,):
		pkeys   = sorted(list(self.pub_dict.keys()))
		for ii in range(len(pkeys)):
			if '[' in pkeys[ii] and ']' in pkeys[ii]:
				
				lft = pkeys[ii].find('[')
				rgt = pkeys[ii].find(']')
				if rgt != lft+1: 															# Remove Redundant Brackets
					oldname   = copy.deepcopy(pkeys[ii])
					newname   = pkeys[ii].replace('[', '')
					newname   = newname.replace(']', '')

					self.pub_dict[newname] = self.pub_dict[oldname]
					del self.pub_dict[oldname]

					pkeys[ii] = newname
					# print('{:4d} {} \t ** CORRECTED **'.format(ii, pkeys[ii], self.pub_dict[pkeys[ii]]['title']))
				try:
					if 'eng' in self.pub_dict[pkeys[ii]]['language']:
						print('{:4d} {:>30}... {}'.format(ii, pkeys[ii][:40], self.pub_dict[pkeys[ii]]['title']))						
				except Exception as e:
						print('{:4d} {:>30}... {}'.format(ii, pkeys[ii], self.pub_dict[pkeys[ii]]['title']))

	def find_abstracts(self,):
		pkeys        = sorted(list(self.pub_dict.keys()))

		for ii in range(len(pkeys)):
			pub = self.pub_dict[pkeys[ii]]
			if 'abstract' not in list(pub):
				print('{:5d} {} {}'.format(ii, pkeys[ii], pub['title'][:40]))

	def find_reviews(self, remove=False):
		pkeys        = sorted(list(self.pub_dict.keys()))

		for ii in range(len(pkeys)):
			pub = self.pub_dict[pkeys[ii]]
			if 'document type' in list(pub):
				if 'review' in pub['document_type']:
					self.review +=1

					if remove == True:
						del self.pub_dict[pkeys[ii]]

		txt = 'Identified as Review'
		if remove == False:
			print('\t{:5d} {:<40} (Did NOT Remove)'.format(self.review, txt))
		else:
			print('\t{:5d} {:<40} (Removed)'.format(self.review, txt))

	def find_years(self, year_thresh=1989, remove=False, plot=False):
		pkeys = sorted(list(self.pub_dict.keys()))
		years = ['{:4d}'.format(ii) for ii in range(1960,2022)]
		ydict = {}

		for ii in range(len(years)):
			ydict[years[ii]] = ii
		
		y     = np.zeros([len(years)])
		for ii in range(len(pkeys)):
			year = pkeys[ii].split('_')[0]
			try:
				yidx     = ydict[year]
				y[yidx] +=1
				
				if int(year) < year_thresh and remove == True:
				# if int(year) != 2021 and remove == True:
					del self.pub_dict[pkeys[ii]]

			except:
				print('Skip {}'.format(pkeys[ii], year))
				del self.pub_dict[pkeys[ii]]

		y_    = np.where(y < 5)[0]

		if plot == True:
			fig, ax = plt.subplots()
			ax.scatter(years, y, s=50, facecolor='royalblue' , edgecolors='k')
			for ii in y_:
				ax.scatter(years[ii], y[ii], s=50, facecolor='darkorange', edgecolors='k')
			plt.xticks(rotation=45)
			plt.show()

	def duplicate_confidence(self,):
		dup_dict    = {} 																	# Dictionary to Evauate Duplicates
		pkeys       = sorted(list(self.pub_dict.keys())) 									# All Publication Keys in self.pub_dict
	 	
		letters     = list(string.ascii_lowercase) 											# Alphabet in Publication Author
		years       = ['{:4d}'.format(ii) for ii in range(1960,2022)]
		words       = [] 																	# Words in Publication Title

		for ii in range(len(pkeys)):
			pub_yr  = pkeys[ii].split('_')[0] 												# Publication Year
			author  = pkeys[ii].split('_')[1] 												# Publication 1st Author
			author  = list(author) 															# Publication 1st Author as Ind. Letter

			pub_tit = self.pub_dict[pkeys[ii]]['title'] 									# Publication Title 
			pub_tit = pub_tit.lower() 														# Publication Title Lower Case
			pub_tit = re.sub('\W+',' ', pub_tit) 											# Publication Title Remove Special Characters
			pub_tit = pub_tit.split(' ') 													# Publication Title as List
			if '' in pub_tit:
				pub_tit.remove('')

			dup_dict[pkeys[ii]] = [author, [pub_yr], pub_tit]
			
			words.extend(pub_tit)

		words       = sorted(list(set(words)))

		indices     = {}
		word_cnt    = 0
		for ii in range(len(letters)):
			indices[letters[ii]] = ii
		
		for ii in range(len(years)):
			indices[years[ii]] = ii + len(letters)

		for ii in range(len(words)):
			if len(words[ii]) > 1 and words[ii].isnumeric() == False:
				indices[words[ii]] = word_cnt + len(letters) + len(years)
				word_cnt          += 1

		self.m      = np.zeros([len(pkeys), (len(letters) + len(years) + word_cnt)])  # Correlation Matrix

		for ii in range(len(pkeys)):
			for jj in range(len(dup_dict[pkeys[ii]][0])):
				if dup_dict[pkeys[ii]][0][jj] in list(indices.keys()):
					idx = indices[dup_dict[pkeys[ii]][0][jj]]
					self.m[ii, idx] +=1

			for jj in range(len(dup_dict[pkeys[ii]][1])):
				if dup_dict[pkeys[ii]][1][jj] in list(indices.keys()):
					idx = indices[dup_dict[pkeys[ii]][1][jj]]
					self.m[ii, idx] +=1

			for jj in range(len(dup_dict[pkeys[ii]][2])):
				if dup_dict[pkeys[ii]][2][jj] in list(indices.keys()):
					idx = indices[dup_dict[pkeys[ii]][2][jj]]
					self.m[ii, idx] +=1

		zeros     = np.where(np.max(self.m, axis=0) == 0) 									# Find Zeros

		mx        = np.max(self.m, axis=0) 													# Max of Each Word Frequency
		mx[zeros] = 1 																		# Find Zero Occurences

		self.m   /= mx[None, :] 															# Normalize Word Frequency

		# idx_words = list(indices.keys())
		# for ii in range(500,600):
		# 	idx   = int(indices[idx_words[ii]])
		# 	print('{:3d} {:5d} {:>12} {}'.format(ii, idx, idx_words[ii], self.m[:15,idx]))

		square    = np.zeros([self.m.shape[0],self.m.shape[0]])

		m         = copy.deepcopy(self.m) 													# Copy Array
		m        -= np.mean(self.m, axis=1)[:,None] 										# Zero-Mean
		m_        = np.sqrt(np.sum(copy.deepcopy(m)**2, axis=1)) 							# Error

		tstart    = t0.time()
		for ii in range(self.m.shape[0]):
			tstart_ = t0.time()
			for jj in range(ii+1, self.m.shape[0]):
				square[ii,jj] = np.matmul(m[ii,:].transpose(), m[jj,:]) / (m_[ii] * m_[jj])			
			
			if ii % 100 == 0:
				if ii % 1000 == 0 or ii == (self.m.shape[0]-1):
					print('\t{:5d} {:6.3f} seconds | (Total = {:7.3f} seconds)'.format(ii, t0.time()-tstart_,t0.time()-tstart))
				else:
					print('\t{:5d} {:6.3f} seconds | (Total = {:7.3f} seconds)'.format(ii, t0.time()-tstart_,t0.time()-tstart), end='\r')

		print('{:2f} - numpy manual'.format(t0.time() - tstart))
		
		a = np.where(square > (np.max(square) * .7))
		b = np.zeros([a[0].shape[0]])

		for ii in range(a[0].shape[0]):
			b[ii] = square[a[0][ii], a[1][ii]]

		b = np.argsort(b)

		for ii in b:
			if square[a[0][ii], a[1][ii]] > .85:
				print('  {:.4f} ** |  {} {}  |  {} | {}'.format(square[a[0][ii], a[1][ii]], pkeys[a[0][ii]], pkeys[a[1][ii]], self.pub_dict[pkeys[a[0][ii]]]['title'][:70], self.pub_dict[pkeys[a[1][ii]]]['title'][:70]))
			else:
				print('  {:.4f}    |  {} {}  |  {} | {}'.format(square[a[0][ii], a[1][ii]], pkeys[a[0][ii]], pkeys[a[1][ii]], self.pub_dict[pkeys[a[0][ii]]]['title'][:70], self.pub_dict[pkeys[a[1][ii]]]['title'][:70]))

		print('\n\t{:5d} Total Potential Duplicates'.format(a[0].shape[0]))

	def write_bibtxt(self,sort_keys=True):
		fname  = '{}/2021_Citations_T2.bib'.format(self.basedir)
		# fname  = '{}/Citations_Year/Citations_Revised_2021.bib'.format(self.basedir)
		if sort_keys == True:
			pkeys     = sorted(list(self.pub_dict.keys()))
		else:
			pkeys     = list(self.pub_dict.keys())

		lcurl  = '{'
		rcurl  = '}'

		bibtxt = open(fname, 'w') 
		for ii in range(len(pkeys)):
			pub    = self.pub_dict[pkeys[ii]]	
			pkeys_ = list(pub.keys()) 

			bibtxt.write('\n@article{}{}'.format(lcurl, pkeys[ii]))
			for jj in range(len(pkeys_)):
				if pkeys_[jj] != 'document_type' and pkeys_[jj] != 'art_number':
					bibtxt.write(',\n    {}={}{}{}'.format(pkeys_[jj], lcurl, pub[pkeys_[jj]], rcurl))
			if 'abstract' not in pkeys_:
				bibtxt.write(',\n    abstract={}{}'.format(lcurl, rcurl))

			bibtxt.write('\n{}\n'.format(rcurl))
		bibtxt.close()

	def clinical_ignore(self,neuro):
		neuro['Acute'                           ] = [0, [], ['Acute', 'Poison', 'Heat Stroke', 'Hypobaric', 'Hypobaria', 'Hypercapnia', 'Altitude', 'CO2']]
		neuro['Ataxia Telangiectasia'           ] = [0, [], ['Ataxia Telangiectasia', 'A-T']]

		neuro['Dementia'                        ] = [0, [], ['Dementia', 'Frontotemporal', 'Frontotemporal Lobar', 'Spongiform', 'Creutzfeldt-Jakob Disease',
															 'Microtubule-Associated Protein Tau', 'MAPT', 'Vanishing White Matter Disease',
															 'Vanishing White Matter', 'Vascular Dementia']]
		neuro['Iron Accumulating'               ] = [0, [], ['c19orf12', 'Chromosome 19 Open Reading Frame 12', 'Mitochondrial Membrane Protein-Associated', 'MPAN']]

		neuro['Auditory'                        ] = [0, [], ['Auditory', 'Tinnitis', 'Hearing', 'Acoustic']]
		neuro['Personality Disorder'            ] = [0, [], ['Personality Disorder', 'Borderline Personality Disorder', 'Borderline Personality', 'Borderline']]
		neuro['Behavior'                        ] = [0, [], ['fMRS', 'functional MRS', 'Memory', 'Learning', 'Reward', 'Intelligence', 'Behavioral Task', 'Aggression', 'Emotion', 'Emotional']]

		neuro['Eating Disorder'                 ] = [0, [], ['Eating Disorder', 'Anorexia', 'Anorexia Nervossa', 'Bulimia', 'Bulimic']]

		neuro['Sexually Transmitted'            ] = [0, [], ['HERPES', 'Syphillis', 'Neurosyphillis', 'Immunodeficiency', 'Human Immunodeficiency Virus', 
															 'Acquired Immunodeficiency Syndrome', 'AIDS', 'Leukoencephalopathy']]
		neuro['Meningitis'                      ] = [0, [], ['Meningitis']]
		neuro['Encephalitis'                    ] = [0, [], ['Encephalitis', 'Encephalopathy']]
		neuro['Hydrocephalus'                   ] = [0, [], ['Hydrocephalus']] 
		neuro['Lyme Neuroborreliosis'           ] = [0, [], ['Lyme Neuroborreliosis', 'Lyme', 'Borrelia Burgdorferi', 'Borrelia', 'Burgdorferi']]

		neuro['Wilson\'s Disease'               ] = [0, [], ['Wilson\'s Disease']]
		neuro['Prader-Willi'                    ] = [0, [], ['Prader-Willi']]
		neuro['Sturge-Weber'                    ] = [0, [], ['Sturge-Weber']]
		neuro['Focal Cortical Dysplasia'        ] = [0, [], ['Focal Cortical Dysplasia']]
		neuro['West'                            ] = [0, [], ['West', 'West Syndrome']]
		neuro['Tuberous Sclerosis'              ] = [0, [], ['Tuberous Sclerosis', 'Tuberous Sclerosis Complex']]
		neuro['Glaucoma'                        ] = [0, [], ['Glaucoma', 'Eye Pressure']]
		neuro['Down Syndrome'                   ] = [0, [], ['Down Syndrome', 'Down\'s syndrome', 'Trisomy', 'Trisomy 21']] 
		neuro['DiGeorge Syndrome'               ] = [0, [], ['DiGeorge Syndrome', '22q11.2', '22q11.2 Deletion']] 
		neuro['Xeroderma Pigmentosum'           ] = [0, [], ['Xeroderma Pigmentosum']]  
		neuro['Pelizaeus-Merzbacher'            ] = [0, [], ['Pelizaeus-Merzbacher']]  

		neuro['MELAS'                           ] = [0, [], ['MELAS', 'Mitochondrial Encephalomyopathy', 'Lactic Acidosis', 'Stroke-Like']] 
		neuro['Acidopathies'                    ] = [0, [], ['Acidopathies', 'Acidopathy', 'Amino Acid', 'Phenylketonuria', 'Hyperargininemia', 'Homocystinuria', 'Tyrosinemia',
		 													 'Methylmalonic Aciduria', 'Aciduria', ]] 
		neuro['Deficiency'                      ] = [0, [], ['Deficiency', 'Guanidinoacetate Methyltransferase', 'Adenylosuccinate Lyase', 'Adenosine Kinase', 
																				  'Creatine Transport Deficiency', '3-Methylcrotonylglycinuria', 'Carbamoyltransferase']] 
		neuro['Lipid Storage'                   ] = [0, [], ['Gaucher\'s', 'Niemann-Pick', 'Chanarin-Dorfman', 'Zellweger', 'Tay-Sachs', 'Gangliosidosis', 'GM2-Gangliosidosis']] 
		neuro['Maple Syrup Urine'               ] = [0, [], ['Maple Syrup Urine', 'Alopecia', 'Erythroderma']]
		neuro['Leukodystrophy'                  ] = [0, [], ['Leukodystrophy', 'Metachromatic', 'Adrenoleukodystrophy', 'Pelizaeus-Merzbacher', 'Canavan']]

		neuro['Stress'                          ] = [0, [], ['Stress', 'Exhaustion', 'Socioeconomic Deprivation']]
		neuro['Malnutrition'                    ] = [0, [], ['Malnutrion', 'Wernicke Encephalopathy', 'Wernicke\'s Encephalopathy', 'Thiamin Deficiency', 'Thiamine Deficiency', 'Dehydration']]


		neuro['Stress-Management'               ] = [0, [], ['Meditation', 'Tai Chi']]
		neuro['Pain-Management'                 ] = [0, [], ['Acupuncture', 'Massage']]
		neuro['Hypothermia'                     ] = [0, [], ['Hypothermia']]
	
		neuro['Type 1 Diabetes'                 ] = [0, [], ['Type 1 Diabetes', 'T1D']]
		neuro['Steroid'                         ] = [0, [], ['Steroid', 'Testosterone', 'Anabolic', 'Androgenic', 'Cortisol']]
		neuro['Fetal'                           ] = [0, [], ['Fetal', 'Fetus',  'Gestational']]

		neuro['Cancer'                          ] = [0, [], ['Tumor', 'Glioma', 'Blastoma', 'Cytoma', 'Lymphoma', 'IDH1', 'IDH2', 'Immunoglobulin Heavy Locus']]
		neuro['Treatment'                       ] = [0, [], ['Radiation', 'Chemotherapy', 'Resection']]
		
		neuro['Non-Neurological'                ] = [0, [], ['Liver', 'Cirrhosis', 'Breast', 'Hepatic', 'Hepatic Encephalopathy', 'Epstein-Barr', 'Blindness',
															 'Hepatitis', 'Hepatitis C', 'Hepatitis-C', 'HCV', 'Eisenmenger', 'Primary Biliary Cholangitis', 'Aneurysm',
															 'Osteoarthritis', 'Arthritis', 'Cervical Myelopathy', 'Hyperthyroidism', 'Thyroid']]

		neuro['Musclur Dystrophy'               ] = [0, [], ['Musclur Dystrophy', 'Duchenne', 'Becker', 'Limb-Girdle', 'Myotonic']]
		neuro['General Movement'                ] = [0, [], ['Ataxia', 'Paralysis', 'Movement', 'Motor Cortex', 'Palsy', 'Spinocerebellar', 'Spinocerebellar Ataxias']]
		return neuro

	def clinical_groups(self,):

		neuro = {}
		# neuro['Healthy'                         ] = [0, [], ['Control']]
		neuro['Reproducibility'                 ] = [0, [], ['Fitting', 'Algorithm', 'Reproducibility', 'Repeatability', 'Reliability']]  
		neuro['Region'                          ] = [0, [], ['Region', 'Location']]
		neuro['Sex Differences'                 ] = [0, [], ['Sex Differences', 'Gender']]
		neuro['Field Strength'                  ] = [0, [], ['Field Strength', '4 Tesla', '4T', '4.7T', '4.7 Tesla', '7 Tesla' '7T', '9.4 Tesla' '9.4T']]
		neuro['Metabolite'                      ] = [0, [], ['Glutathione', 'GABA', 'NAAG', 'Aspartic Acid', 'Aspartate', 'Ascorbic Acid', 'Ascorbate', 'Vitamin C', 
															 'Phosphoethanolamine', 'Serine', 'Taurine', 'Alanine', 'Glucose', 'Threonine', 'Lactic Acid', 'Lactate',
														     'Valine', '2-Hydroxyglutarate', 'Beta-hydroxybutyrate', 'Betahydroxybutyrate', 'Hydroxy-butyrate',
														     '3-hydroxybutyrate', 'Ketone', 'Glycine',  'Scyllo', 'Scylloinositol', 'Homocarnosine', 'NAD+',
															 'NAD(+)', 'Nicotinamide Adenine Dinucleotide', 'Cystathionine']]
		neuro['Macromolecules'                  ] = [0, [], ['Macromolecules', 'Macromolecule', 'MM09', 'MM12', 'MM14', 'MM17', 'MM17',
															 'MM18', 'MM19', 'MM20', 'MM21', 'MM22', 'MM23', 'MM25', 'MM27', 'MM29', 'MM30',
															 'MM32', 'MM34', 'MM35', 'MM36', 'MM37', 'MM38', 'MM39', 'MM42']]
		neuro['Adolescent'                      ] = [0, [], ['Child', 'Children', 'Youth']]
		neuro['Stroke'                          ] = [0, [], ['Stroke', 'Infarction', 'Ischemia', 'Ischemic', 'Hemorrhagic', 'Aphasia', 'Transient Ischemic Attack']]
		neuro['Traumatic Brain Injury'          ] = [0, [], ['Traumatic Brain Injury', 'TBI', 'Concussion', 'Mild Traumatic Brain Injury',
		 													 'Chronic Traumatic Encephalopathy', ]]

		neuro['Alzheimer\'s Disease'            ] = [0, [], ['Alzheimer\'s Disease' , 'Alzheimer', 'Alzheimers', 'Late-Onset Alzheimer']]
		neuro['Parkinson\'s Disease'            ] = [0, [], ['Parkinson\'s Disease' , 'Parkinson', 'Parkinsons', 'Parkinsonism', 'L-Dopa']]
		neuro['Huntington\'s Disease'           ] = [0, [], ['Huntington\'s Disease', 'Huntington', 'Huntingtons', 'Chorea']]
		neuro['Amyotrophic Lateral Sclerosis'   ] = [0, [], ['Amyotrophic', 'Amyotrophic Lateral Sclerosis', 'Lou Gehrig']]
		neuro['Multiple Sclerosis'              ] = [0, [], ['Multiple Sclerosis', 'Radiologically Isolated Syndrome', 'Remitting Multiple Sclerosis', 
															 'Relapsing-Remitting Multiple Sclerosis', 'Relapsing Multiple Sclerosis']] 
		neuro['Mild Cognitive Impairment'       ] = [0, [], ['Mild Cognitive Impairment', 'Amnestic Mild Cognitive Impairment', 'Cognitive Impairment']]

		neuro['Schizophrenia'                   ] = [0, [], ['Schizophrenia', 'Psychosis', 'Hallucination', 'Positive Symptoms', 'Negative Symptoms', 'PANSS']]
		neuro['Epilepsy'                        ] = [0, [], ['Epilepsy', 'Epileptic', 'Temporal Lobe Epilepsy', 'Seizure', 'Grand Mal', 
															 'Petite Mal', 'Sildenafil', 'Gabapentin', 'Levetiracetam']]
		neuro['Migraine'                        ] = [0, [], ['Migraine', 'Migraineurs', 'Migraineur', 'Aura', 'Headache']]
		neuro['Pain'                            ] = [0, [], ['Pain', 'Pain-Management', 'Fibromyalgia', 'Chronic Pain', 'Weakness']]
		neuro['Sleep'                           ] = [0, [], ['Sleep', 'Sleep Quality', 'Apnea', 'Sleep Apnea', 'Bruxism', 'Insomnia']]
		neuro['Obsessive Compulsive Disorder'   ] = [0, [], ['Obsessive', 'Compulsive', 'Obsessive-Compulsive', 'Obsessive-Compulsive Disorder',  'OCD']]
		neuro['Addiction'                       ] = [0, [], ['Addiction', 'Substance Abuse', 'Tabacco', 'Smoking', 'Nicotine', 'Alcohol', 'Amphetamine',
															 'Methamphetamine', 'Methadone',  'Heroin', 'Opiate', 'Cannabis', 'Marijuana', 
															 'Abstinence', 'Abstinent', 'Cocaine', 'THC', 'Dependence']]
		neuro['Autism'                          ] = [0, [], ['Autism', 'Autism Spectrum Disorder', 'Autistic', 'Asperger', 'Asperger\'s', 'Sensory Over-Responsivity',]]
		neuro['Attention Deficit'               ] = [0, [], ['Attention-Deficit', 'Attention Deficit', 'Attention-Deficit Disorder', 'Attention-Deficit Hyper-Activity', 'Hyper-Activity', 'Hyper Activity', 'ADHD']]
		neuro['Tourettes'                       ] = [0, [], ['Tourettes', 'Tourettes Syndrome']]

		neuro['Bipolar'                         ] = [0, [], ['Bipolar', 'Bipolar Disorder', 'Bipolar-I', 'Bipolar I', 'Bipolar-1', 'Unipolar', 'Anhedonia', 'Mania', 'Manic']]
		neuro['Anxiety'                         ] = [0, [], ['Anxiety', 'Panic Attack', 'Worry', 'Neuroticism', 'Generalized Anxiety Disorder', 'Anxiety Disorder']]
		neuro['Depression'                      ] = [0, [], ['Depression', 'Depressive', 'Depressive Disorder', 'Suicidal', 'Major Depressive Disorder']]
		neuro['Post-Traumatic Stress Disorder'  ] = [0, [], ['Post-Traumatic Stress Disorder', 'Post Traumatic Stress Disorder', 'PTSD']]

		neuro['Type 2 Diabetes'                 ] = [0, [], ['Type 2 Diabetes', 'T2D', 'Insulin Resistant', 'Insulin Sensitivity']] 

		neuro['Menstrual'                       ] = [0, [], ['Menstrual', 'Menopause', 'Peri-Menopause', 'Menipausal', 'Estrogen', 'Estradiol', 'Progesterone', 'Premenstrual Dysphoric Disorder',  'Postpartum', 'Peripartum']]
		neuro['Polypeptide'                     ] = [0, [], ['Oxytocin', 'Growth Hormone', 'Kisspeptin']]
		
		neuro['Infant'                          ] = [0, [], ['Infant', 'Newborn', 'Neonate', 'Neonatal', 'Preterm', 'Premature', 'Bilirubin Encephalopathy']]

		neuro['Aging'                           ] = [0, [], ['Aging', 'Lifespan', 'Normal Aging', 'Healthy Aging', '80 and over',]]
		neuro['APOE'                            ] = [0, [], ['APOE', 'APOE3', 'APOE4', 'Lipoprotein', 'Apolipoprotein', 'Apolipoprotein E']]
		neuro['Obesity'                         ] = [0, [], ['Obesity', 'Body Mass Index', 'Body-Mass', 'BMI', 'Overweight',]]
		neuro['Arterial'                        ] = [0, [], ['Atherosclerosis', 'Coronary Artery Disease', 'Cholesterol']]
		
		neuro['Lifestyle'                       ] = [0, [], ['Diet', 'Caloric Restriction', 'Fasting', 'Fasted', 'Intermittent Fasting', 'Ketogenic', 'Low Carbohydrate',
															 'Long-Chain Polyunsaturated Fatty Acid', 'LCPUFA', 'Supplement', 'Supplementation,',
															 'Vitamin', 'Mineral', 'Minerals', 'Precursor', 'Thiamine', 'Riboflavin', 'Niacin', 'Pantothenic Acid', 
															 'Pyridoxine', 'Biotin', 'Folic Acid', 'Folate', 'Cobalamin', 'Calciferol', 'Nicotinamide', 'Nicotinamide Mononucleotide',
															 'Nicotinamide Adenine Dinucleotide', 'NAD+', 'Creatine Monohydrate', 'Supplementation',
															 'Exogenous', 'Exercise', 'Physical Activity', 'Athlete', 'Running', 'Swimming']] 
		neuro['Drug'                            ] = [0, [], ['Drug', 'Ketamine', 'Clozapine', 'Isoniazid', 'N-Acetylcysteine', 'Lamotrigine', 'Antidepressant',
															 'Anesthetic', 'Anesthetia', 'Natalizumab', 'Minocycline', 'Inhibitor', 'Antiepileptic', 'Antipsychotic',
															 'Citalopram']]
		neuro['Stimulation'                     ] = [0, [], ['Stimulation', 'Electroconvulsive', 'Electroconvulsive Therapy', 'TDCS', 'Transcranial Direct Current Stimulation',
														     'TMS', 'Transcranial Magnetic Stimulation', 'Repetitive Transcranial Magnetic Stimulation', 'rTMS',]]

		neuro = self.clinical_ignore(neuro)

		degn = ['Alzheimer\'s Disease', 'Parkinson\'s Disease', 'Huntington\'s Disease', 'Amyotrophic Lateral Sclerosis', 'Multiple Sclerosis', 'Mild Cognitive Impairment']
		psyc = ['Schizophrenia', 'Epilepsy', 'Migraine', 'Pain', 'Sleep', 'Obsessive Compulsive', 'Addiction']
		mood = ['Bipolar', 'Depression', 'Anxiety', 'Post-Traumatic Stress Disorder']
		trau = ['Traumatic Brain Injury', 'Stroke', 'Acute']
		risk = ['Aging', 'APOE', 'Obesity', 'Type 2 Diabetes', 'Stress']
		intv = ['Diet', 'Exercise', 'Drug', 'Electroconvulsive', 'TMS', 'TDCS']
		adol = ['Adolescent', 'Autism', 'Attention-Deficit', 'Tourettes']

		subcats   = {'degn': [0, degn, []],
					 'psyc': [0, psyc, []],
					 'mood': [0, mood, []],
					 'trau': [0, trau, []],
					 'risk': [0, risk, []],
					 'intv': [0, intv, []],
					 'adol': [0, adol, []]}

		skeys     = list(subcats.keys())

		pkeys     = sorted(list(self.pub_dict.keys())) 										# Publications
		nlist     = sorted(list(neuro.keys()))

		cats      = np.zeros([len(nlist), len(nlist)])
		pcnt      = np.zeros([len(nlist), len(pkeys)])
		ncnt      = np.zeros([len(nlist)])

		for jj in range(len(skeys)):
			for kk in range(len(nlist)):
				if nlist[kk] in subcats[skeys[jj]][1]:
					subcats[skeys[jj]][2].append(kk)

		for jj in range(len(nlist)):
			for ii in range(len(pkeys)):
				pub   = self.pub_dict[pkeys[ii]]

				if 'keywords' in list(pub.keys()):
					ptext = ''.join((pub['title'], pub['keywords'], pub['abstract']))
				else:
					ptext = ''.join((pub['title'], pub['abstract']))

				ptext   = ptext.lower()
				wrd_cnt = 0
				wlist   = neuro[nlist[jj]][-1]

				for kk in range(len(wlist)):
					wrd_cnt += ptext.count(wlist[kk].lower())
				
				if wrd_cnt > 1:
					# print('{:3d} {:4d}'.format(int(ncnt[jj]), ii), '{:<20}'.format(nlist[jj]), '{:3d}'.format(wrd_cnt))
					ncnt[jj]    += 1
					pcnt[jj,ii] += wrd_cnt

		nlabmx = np.zeros([len(nlist)])
		for ii in range(len(pkeys)):
			a       = np.where(pcnt[:,ii] > 1)[0]
			string  = ''
			labels  = {} 

			if a.shape[0]  >= 1:
				for jj in range(a.shape[0]):
					string  = '{}{} '.format(string , nlist[a[jj]]) 
					labels[nlist[a[jj]]] = pcnt[a[jj],ii]

					if pcnt[a[jj],ii] > nlabmx[a[jj]]:
						nlabmx[a[jj]] = pcnt[a[jj],ii]

			self.pub_dict[pkeys[ii]]['labels'] = labels
			# print('{:3d} {:<20} {} {}'.format(ii, pkeys[ii], a, string))
			if a.shape[0] == 1:
				cats[a[0],a[0]] +=1
			elif a.shape[0] > 1:
				b = np.argsort(pcnt[a,ii])

				cats[a[b[-1]],a[b[-2]]] +=1
				cats[a[b[-2]],a[b[-1]]] +=1

		olist = np.argsort(ncnt)[::-1]
		ocnt  = 0 
		for jj in range(olist.shape[0]):
			ocnt += int(ncnt[olist[jj]])
			# print('{:<30} {:3d}  {:4d}'.format(nlist[olist[jj]][:30], int(ncnt[olist[jj]]), ocnt))

		diag  = 0
		off   = 0
		for ii  in range(len(nlist)):
			for jj in range(ii, len(nlist)):
				if ii == jj:
					diag += cats[ii,jj]
					# print('{:<30} {:3d}'.format(nlist[ii], int(np.ceil(cats[ii,jj])) ))
				else:
					off  += cats[ii,jj]

		scats = np.zeros([len(skeys), len(skeys)])
		for ii in range(cats.shape[0]):
			for jj in range(cats.shape[1]):

				for kk in range(len(skeys)):
					if nlist[ii] in subcats[skeys[kk]][1]:
						iarg = kk
					if nlist[jj] in subcats[skeys[kk]][1]:
						jarg = kk

				scats[iarg, jarg] += cats[ii,jj]
				# print('{} {} {} {} {}'.format(nlist[ii], nlist[jj], int(cats[ii,jj]), skeys[iarg], skeys[jarg]))


		print(scats, '\n\n')
		# print(len(nlist))
		# print(cats[:15,:15])

		print('Diagonal     = {}'.format(diag))
		print('Off-Diagonal = {}\n'.format(off/2))


		for ii in range(len(pkeys)):
			# print(self.pub_dict[pkeys[ii]]['labels'])
			labs  = list(self.pub_dict[pkeys[ii]]['labels'].keys())
			vals  = np.zeros([len(labs)], dtype=np.int16)
			a     = []
			for jj in range(len(labs)):
				a.append(nlist.index(labs[jj]))
				vals[jj]  = int(self.pub_dict[pkeys[ii]]['labels'][labs[jj]])
				# vals[jj] /= nlabmx[a[jj]]

			srt   = np.argsort(vals)[::-1]
			new   = {}
			for jj in range(srt.shape[0]):
				new[labs[srt[jj]]] = vals[srt[jj]]

			self.pub_dict[pkeys[ii]]['labels'] = copy.deepcopy(new)

			print('{:4d} {:<25}'.format(ii, pkeys[ii]), self.pub_dict[pkeys[ii]]['labels'])

		print('\n\n')
		# print(self.pub_dict[pkeys[ii]].keys())
		# fig,ax = plt.subplots()
		# ax.imshow(scats/np.max(scats, axis=0), cmap='Greys', interpolation='nearest')
		# ax.set_yticks(np.arange(7))
		# ax.set_yticklabels(skeys)
		# ax.set_xticks(np.arange(7))
		# ax.set_xticklabels(skeys)
		# plt.show()		

	def write_csv(self,):
		pkeys  = sorted(list(self.pub_dict.keys()))
		cols   = ['name'   , 'year' , 'lab1'  , 'lab2' , 'lab3' , 'doi',
		 		  'author' , 'title', 'volume', 'journ', 'month', 'url',
		 		  'keyword',                                           ]

		name   = []
		year   = []
		lab1   = []
		lab2   = []
		lab3   = []
		doi    = []
		author = []
		title  = []
		volume = []
		journ  = []
		url    = []
		keywrd = []
		dwnld  = []

		pdir   = 'C:/Users/agudm/Desktop'
		papers = glob.glob('{}/NMR_T2/*'.format(pdir,))
		for ii in range(len(papers)):
			papers[ii] = papers[ii].split('\\')[-1][:-4]

		for ii in range(len(pkeys)):
			if pkeys[ii] in papers:
				dwnld.append('X')
			else:
				dwnld.append('')

			name.append(pkeys[ii])
			year.append(  self.pub_dict[pkeys[ii]]['year'    ])
			try:
				doi.append(   self.pub_dict[pkeys[ii]]['doi' ])
			except Exception as e:
				pass

			author.append(self.pub_dict[pkeys[ii]]['author'  ])
			title.append( self.pub_dict[pkeys[ii]]['title'   ])
			volume.append(self.pub_dict[pkeys[ii]]['volume'  ])
			journ.append( self.pub_dict[pkeys[ii]]['journal' ])

			if 'url' in list(self.pub_dict[pkeys[ii]].keys()):
				url.append(   self.pub_dict[pkeys[ii]]['url'    ])
			else:
				url.append('')
			
			if 'keywords' in list(self.pub_dict[pkeys[ii]].keys()):
				keywrd.append(self.pub_dict[pkeys[ii]]['keywords'])
			else:
				keywrd.append('')


			labs   = list(self.pub_dict[pkeys[ii]]['labels'].keys())
			if len(labs) > 0:
				lab1.append(labs[0])
			else:
				lab1.append('')
			if len(labs) > 1 and self.pub_dict[pkeys[ii]]['labels'][labs[1]] > 2:
				lab2.append(labs[1])
			else:
				lab2.append('')
			if len(labs) > 2 and self.pub_dict[pkeys[ii]]['labels'][labs[2]] > 2:
				lab3.append(labs[2])				
			else:
				lab3.append('')

		df = pd.DataFrame({'Name'    : name,
						   'Download': dwnld,
						   'Year'    : year,
						   'Label1'  : lab1,
						   'Label2'  : lab2,
						   'Label3'  : lab3,
						   'DOI'     : doi,
						   'Authors' : author,
						   'Title'   : title,
						   'Volume'  : volume,
						   'Journal' : journ,
						   'URL'     : url,
						   'Keyword' : keywrd})

		df.to_csv('{}/NMR_T2/T2_Articles_Download.csv'.format(pdir))
		print(df.head(150))

	def comparison_csv(self,):

		self.total = len(list(self.pub_dict.keys()))

		pkeys      = sorted(list(self.pub_dict.keys()))

		name       = []
		year       = []
		volume     = []
		author     = []
		doi        = []
		title      = []

		for ii in range(len(pkeys)):
			pub = self.pub_dict[pkeys[ii]]

			name.append(pkeys[ii])
			year.append(pub['year'])
			volume.append(pub['volume'])
			author.append(pub['author'])
			title.append(pub['title'])
			try:
				doi.append(pub['doi'])
			except:
				doi.append('')

		df = pd.DataFrame({
						  'Name'  : name,
						  'Year'  : year,
						  'Volume': volume,
						  'Author': author,
						  'DOI'   : doi,
						  'Title' : title,
						  })

		df.to_csv('{}/2022_All_Concentrations_Citations_simplified.csv'.format(self.basedir))


if __name__   == '__main__':
	
	basedir    = 'C:/Users/agudm/Downloads'
	# basedir    = 'G:/Shared drives/Stark Lab/Aaron/Papers/Systematic_Review'
	#basedir    = 'C:/Users/agudm/Desktop'
	# basedir    = 'C:/Users/agudm/Desktop/Spectroscopy_Articles_2017-2021'

	fnames     = [
				  # Relaxation
				  # '2021_WebOfS_T2_Relaxation_Citations.bib',  # 222 Articles
				  # '2022_WebOfS_T2_Relaxation_Citations.bib',    # 235 Articles **
				  # '2021_PubMed_T2_Relaxation_Citations.csv'   # 287 Articles
				  # '2022_PubMed_T2_Relaxation_Citations.csv', 	# 384 Articles **
				  # '2022_SCOPUS_T2_Relaxation_Citations.csv'     # 231 Articles **
				  # '2022_SCOPUS_T2_Relaxation_Citations_Simple.csv'     # 231 Articles **
				  												# 850 Articles Total **


				  # '2021_Citations_T2.bib' 				 	    # 286 Articles + 4 RandomGoogle
				  # Concentration
				  # 'WOS_PubMed_Combined_Original.bib', 		# Original PubMed + WOS = 6741 (789 > 2017)
				  # 'Citations_Revised_Remove.bib'
				  'WOS_PubMed_Combined_Remove.bib'  , 		# PubMed = 5470;  WOS = 3328
				  'Scopus_Search_1708.bib',
				  # '2021_2022_PubMed_Search_Concentrations.csv',
				  '2021_2022_PubMed_Final_Concentrations.csv' # PubMed = 190 of 231 (Completed 2021-2022);
				  # 'PsycArticles_Database_Remove.csv', 		# Psyc   = 171
				  # 'Scopus_Search_1708.bib'   , 				# Scopus = 1708
				  # 'Scopus_Search_1708_Remove.bib'   , 		# Scopus = 1708
				  												# Total Articles = 10,677 Articles
				 ]

	pub        = CombinePublications(basedir)

	pub.read_articles(fnames)
	pub.collapse_duplicates(display=True)
	pub.find_conferences(remove=True)
	pub.find_non_english(remove=True)
	# pub.find_no_authors(remove=True)
	# # pub.find_books(remove=True)
	# pub.find_editorials(remove=True)
	# pub.find_reviews(remove=True)
	pub.find_years(year_thresh=2017, remove=True)
	pub.running_total()
	# pub.comparison_csv()

	# # pub.clinical_groups()
	# pub.write_csv()

	# # pub.find_exclusions(remove=True)
	# # pub.write_bibtxt(sort_keys=True)

	# # pub.find_incomplete()

	# remove = ['1992_sappey-marinier_26', '2003_ramani_21', '2000_rose_18', '1986_chatel_3', '2013_marjaaska_8',
	# 		  '2010_ongar_63', '2001_mlynrik_14', '2016_antoniohernandez-tamames_26', '2009_fleysher_27a', '2015_erdelyi-botor_55',
	# 		  '2006_de_56', '1986_majumdar_3b', '1983_huk_1']
	# for ii in range(len(remove)):
	# 	del pub.pub_dict[remove[ii]]

	pub.duplicate_confidence()
	pub.running_total()

	pkeys      = sorted(list(pub.pub_dict.keys()))
	# for ii in range(len(pkeys)):
	# 	print(pkeys[ii])

	print('\n\n')
	# print(pub.path0n)
	# print(pub.path1n)
	# print(pub.path2n)

	# print(pub.duplicate)
	print('\nErrors:')
	print(pub.errors)
	# print('Warnings:')
	# print(pub.warnings)

	# cnt      = 0
	# pub_keys = list(pub.pub_dict.keys()) 
	# for ii in range(len(pub_keys)):
	# 	pub_dict = pub.pub_dict[pub_keys[ii]]

	# 	year     = pub_dict['year']
	# 	if int(year) > 2016 and int(year) < 2018:
			
	# 		abst = pub_dict['abstract']
			
	# 		if 'child' in abst:
	# 			cnt +=1 
	# 			print('{:3d} {:<26} {:<160}'.format(cnt, pub_keys[ii], pub_dict['title'][:160]))


	# 	# print('\n{} {}\n'.format(pub_keys[ii], type(pub_keys[ii])))
	# 	# pub_dict = pub.pub_dict[pub_keys[ii]]
	# 	# pkeys    = list(pub_dict.keys())
	# 	# for jj in range(len(pkeys)):
	# 	# 	print('{:2d}'.format(ii), '{:2d}'.format(jj), pkeys[jj], pub_dict[pkeys[jj]])
