## mrs-database:
This repository hosts an open-source database for <i>in vivo</i> <sup>1</sup>H Magnetic Resonance Spectroscopy (MRS) of the brain. Users can download, use, and contribute their own results to this database.

This database was originally created for a <b>Meta-analysis</b> investigating the standard ranges of metabolite concentrations and relaxation times in the <b>Healthy</b> and different <b>Clinical</b> brain. 

For more information, see the [Publication at Analytical Biochemistry](https://doi.org/10.1016/j.ab.2023.115227) or read the Abstract below.

<hr style="height:5px; visibility:hidden;"/>

## Publication Abstract:
Proton (<sup>1</sup>H) Magnetic Resonance Spectroscopy (MRS) is a non-invasive tool capable of quantifying brain metabolite concentrations in vivo. Prioritization of standardization and accessibility in the field has led to the development of universal pulse sequences, methodological consensus recommendations, and the development of open-source analysis software packages. One on-going challenge is methodological validation with ground-truth data. As ground-truths are rarely available for <i>in vivo</i> measurements, data simulations have become an important tool. The diverse literature of metabolite measurements has made it challenging to define ranges to be used within simulations. Especially for the development of deep learning and machine learning algorithms, simulations must be able to produce accurate spectra capturing all the nuances of <i>in vivo</i> data. Therefore, we sought to determine the physiological ranges and relaxation rates of brain metabolites which can be used both in data simulations and as reference estimates. Using the Preferred Reporting Items for Systematic reviews and Meta-Analyses (PRISMA) guidelines, we've identified relevant MRS research articles and created an open-source database containing methods, results, and other article information as a resource. Using this database, expectation values and ranges for metabolite concentrations and T<sub>2</sub> relaxation times are established based upon a meta-analyses of healthy and diseased brains.

<hr style="height:5px; visibility:hidden;"/>

## Citation:
If you use this database, please cite as:
><b>Gudmundson, A. T.</b>, Koo, A., Virovka, A., Amirault, A. L., Soo, M., Cho, J. H., Oeltzschner G., Edden R. A. E., & Stark, C. E. (2023). Meta-analysis and open-source database for in vivo brain Magnetic Resonance spectroscopy in health and disease. Analytical biochemistry, 676, 115227.

<hr style="height:5px; visibility:hidden;"/>

## Keywords:
Human brain, Database, Meta-analysis, Proton MRS, In vivo

<hr style="height:5px; visibility:hidden;"/>

## What's Inside:
<ol>
	<h3><li>CSV files</li></h3>
	<ul>
  		<li><b>References_Concentrations.csv</b></li>
		<p>
		A .csv file detailing all of the information about each of the concentration			articles included in the database. Information (<i>described in more detail 
		below</i>) includes the reference name IDs, study parameters, participant 
		information, acquisition parameters, and analysis parameters.
		</p>
  	</ul>
	<ul>
  		<li><b>Values_Concentrations.csv</b></li>
		<p>
		A .csv file containing the concentrations values corresponding to each article
		listed in the References_Concentrations.csv.
		</p>
  	</ul>
	<ul>
  		<li><b>References_T2.csv</b></li>
		<p>
		A .csv file detailing all of the information about each of the T<sub>2</sub>			articles included in the database. Information (<i>described in more detail 
		below</i>) includes the reference name IDs, study parameters, participant 
		information, acquisition parameters, and analysis parameters.
		</p>
  	</ul>
	<ul>
  		<li><b>Values_T2.csv</b></li>
		<p>
		A .csv file containing the T<sub>2</sub> values corresponding to each article
		listed in the References_T2.csv.
		</p>
  	</ul>
  	
	<h3><li>Python Scripts</li></h3>
		<ul>
	  		<li><b>combine_pubs.py</b></li>
			<p>
			Used to organize, screen, review, and manage publications</p>
	  	</ul>
	  	<ul>
	  		<li><b>supp_functions.py</b></li>
			<p>
			Python functions to perform meta-analysis operations - 
			used in the Jupyter Notebooks below.</p>
	  	</ul>
	  	<ul>
	  		<li><b>Read_Figure.py</b></li>
			<p>
			This was the first version of a Python program that allows you to extract
			the raw data from a figure. This was used when articles did not give the
			raw data values despite including a figure of the data. The <b>current</b> 
			version of this software now lives on <a href="https://github.com/agudmundson/Figure_Reader">https://github.com/agudmundson/Figure_Reader</a></p>
	  	</ul>
  	
	<h3><li>Jupyter Notebooks</li></h3>
		<ul>
	  		<li><b>Analysis_00_Confirm_Supp_Fx.ipynb</b></li>
			<p>
			A Jupyter notebook that demonstrates the Supplemental Meta-analysis 
			tools used within the analyses</p>
	  	</ul>
	  	<ul>
	  		<li><b>Analysis_01_Healthy_Groups.ipynb</b></li>
			<p>
			A Jupyter notebook with the brain metabolite concentration analysis 
			in <b>Healthy</b> Populations.</p>
	  	</ul>
		<ul>
	  		<li><b>Analysis_02_Control_Groups.ipynb</b></li>
			<p>
			A Jupyter notebook with the brain metabolite concentration analysis 
			in <b>Clinical</b> Populations</p>
	  	</ul>
		<ul>
	  		<li><b>Analysis_04_T2_Groups.ipynb</b></li>
			<p> 
			A Jupyter notebook with the brain metabolite relaxation analysis. 
			Here, a meta-regression was developed that can predict T2 relaxation 
			given the experimental conditions</p>
	  	</ul>
	<h3><li>Figures</li></h3>
		<ul>
			<li><b>Figures</b> Directory (Figures from the publication)</li>
			<ul>
		  		<li>2023_Gudmundson_Meta_Figure_01_PRISMA.png</li>
		  		<li>2023_Gudmundson_Meta_Figure_02_Healthy_Concentrations.png</li>
		  		<li>2023_Gudmundson_Meta_Figure_03_Clinical_Concentrations.png</li>
		  		<li>2023_Gudmundson_Meta_Figure_04_T2_Model.png</li>
		  	</ul>
	  	</ul>
	<h3><li>Results</li></h3>
		<ul>
			<li><b>Results</b> Directory (Results from the publication)</li>
			<ul>
		  		<li>Results_Concentration_Clinical.csv</li>
		  		<li>Results_Healthy_All_Groups.csv</li>
		  	</ul>
	  	</ul>
</ol>

## Using the References and Values CSVs:
The References and Values CSVs are to be used together. The references csv holds all of the 
information about a given article, from study details to which analysis were performed. It's important to note that for each reference, there may be multiple entries (<b>ID</b>) if there were results given for multiple experiments (<i>i.e., different brain regions, different pulse sequences, etc.</i>).
 
<ul>
	<h3><li>References CSVs:</li></h3>
	<table>
		<tr>
			<th>Column Name</th>
			<th>Category   </th>
			<th>Datatype   </th>
			<th>Description</th>
		</tr>
		<tr>
			<td>Complete   </td>
			<td>           </td>
			<td>Binary     </td>
			<td>Whether the article has been fully added to the database</td>
		</tr>
		<tr>
			<td>Download    </td>
			<td>            </td>
			<td>Binary      </td>
			<td>Was used internally showing an article was downloaded</td>
		</tr>
		<tr>
			<td>ID          </td>
			<td>            </td>
			<td>Integer     </td>
			<td>An ID for each sub-experiment performed</td>
		</tr>
		<tr>
			<td>Control_ID  </td>
			<td>            </td>
			<td>Integer     </td>
			<td>For studies with multiple groups, this is the corresponding
									      control group's ID for this experimental group. If there 
									      was no control group or this ID is the Control Group, then
									      'No Control' is used.</td>
		</tr>
		<tr>
			<td>Treatment    </td>
			<td>Study        </td>
			<td>Binary       </td>
			<td>Indicates this was a group treated with an intervention.</td>
		</tr>
		<tr>
			<td>Group        </td>
			<td>Study        </td>
			<td>String       </td>
			<td>Healthy or type of clinical population</td>
		</tr>
		<tr>
			<td>Additional_Group_Description</td>
			<td>Study        </td>
			<td>String       </td>
			<td>More information about the given population</td>
		</tr>
		<tr>
			<td>Visit        </td>
			<td>Study        </td>
			<td>String       </td>
			<td>If longitudinal study, this indicates which study visit or session number</td>
		</tr>
		<tr>
			<td>Intervention</td>
			<td>Study        </td>
			<td>String       </td>
			<td>If intervention was performed, this details what the intervention was</td>
		</tr>
		<tr>
			<td>DOI          </td>
			<td>Publication  </td>
			<td>String       </td>
			<td>Digital Object Identifier for the given article</td>
		</tr>
		<tr>
			<td>Title        </td>
			<td>Publication  </td>
			<td>String       </td>
			<td>Title of the article</td>
		</tr>
		<tr>
			<td>Tesla        </td>
			<td>Acquisition  </td>
			<td>Float        </td>
			<td>Field Strength in Tesla of MRI scanner</td>
		</tr>
		<tr>
			<td>Scanner_Manufacturer</td>
			<td>Acquisition  </td>
			<td>String       </td>
			<td>The manufacturer of the scanner</td>
		</tr>
		<tr>
			<td>Scanner_Model</td>
			<td>Acquisition  </td>
			<td>String       </td>
			<td>The model of the scanner</td>
		</tr>
		<tr>
			<td>N_Exclusions</td>
			<td>Study       </td>
			<td>Integer     </td>
			<td> </td>
		</tr>
		<tr>
			<td>N_Total</td>
			<td>Study      </td>
			<td>Integer     </td>
			<td>Total number of subjects</td>
		</tr>
		<tr>
			<td>Female</td>
			<td>Study      </td>
			<td>Integer     </td>
			<td>Total number of female subjects</td>
		</tr>
		<tr>
			<td>Male</td>
			<td>Study      </td>
			<td>Integer     </td>
			<td>Total number of male subjects</td>
		</tr>
		<tr>
			<td>Age        </td>
			<td>Study      </td>
			<td>Float      </td>
			<td>Mean Age of subjects in years</td>
		</tr>
		<tr>
			<td>Age_Std    </td>
			<td>Study      </td>
			<td>Float      </td>
			<td>Standard Deviation for Age of subjects in years</td>
		</tr>
		<tr>
			<td>Water_Suppresion</td>
			<td>Acquisition</td>
			<td>String     </td>
			<td>Type of Water Suppression Used (i.e., CHESS, VAPOR, etc.)</td>
		</tr>
		<tr>
			<td>Lipid_Suppression</td>
			<td>Acquisition</td>
			<td>String     </td>
			<td>Type of additional Lipid Suppression Used</td>
		</tr>
		<tr>
			<td>Localization</td>
			<td>Acquisition</td>
			<td>String     </td>
			<td>Localization Method Used (i.e., PRESS, STEAM, LASER, sLASER, etc.)</td>
		</tr>
		<tr>
			<td>Multi-Scan_Method</td>
			<td>Acquisition</td>
			<td>String     </td>
			<td>Name of multi-part of scan (i.e., MEGA)</td>
		</tr>
		<tr>
			<td>Water Reference</td>
			<td>Acquisition</td>
			<td>Binary     </td>
			<td>Yes/No, was a water reference collected</td>
		</tr>
		<tr>
			<td>Bandwidth (Hz)</td>
			<td>Acquisition</td>
			<td>Float     </td>
			<td>Spectral width of acquisition in Hertz</td>
		</tr>
		<tr>
			<td>Vector Size</td>
			<td>Acquisition</td>
			<td>Integer    </td>
			<td>Number of datapoints in acquisition</td>
		</tr>
		<tr>
			<td>N_Averages</td>
			<td>Acquisition</td>
			<td>Integer    </td>
			<td>Number of transients collected</td>
		</tr>
		<tr>
			<td>TR (ms)</td>
			<td>Acquisition</td>
			<td>Float     </td>
			<td>Repetition Time in milliseconds</td>
		</tr>
		<tr>
			<td>N_TR      </td>
			<td>Acquisition</td>
			<td>Integer   </td>
			<td>Number of Repetition Times Used</td>
		</tr>
		<tr>
			<td>TE (ms)</td>
			<td>Acquisition</td>
			<td>Float     </td>
			<td>Echo Time in milliseconds</td>
		</tr>
		<tr>
			<td>N_TR      </td>
			<td>Acquisition</td>
			<td>Integer   </td>
			<td>Number of Echo Times Used</td>
		</tr>
		<tr>
			<td>TI (ms)</td>
			<td>Acquisition</td>
			<td>Float     </td>
			<td>Inversion Time in milliseconds</td>
		</tr>
		<tr>
			<td>N_TI       </td>
			<td>Acquisition</td>
			<td>Integer    </td>
			<td>Number of Inversion Times Used</td>
		</tr>
		<tr>
			<td>Voxel_Region</td>
			<td>Acquisition </td>
			<td>String     </td>
			<td>Brain region where voxel was placed</td>
		</tr>
	

	</table>		
	</ol>
</ul>


<br>Each article is named by the <b>PubYear_1stAuthor_JournalVolume</b></br>
<br>For every article included, there is an assigned index <b>ID</b>.</br>
Reference
</ul>
