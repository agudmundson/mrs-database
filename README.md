## mrs-database:
This repository hosts an open-source database for <i>in vivo</i> <sup>1</sup>H Magnetic Resonance Spectroscopy (MRS) of the brain. Users can download, use, and contribute their own results to this database.

This database was originally created for a <b>Meta-analysis</b> investigating the standard ranges of metabolite concentrations and relaxation times in the <b>Healthy</b> and different <b>Clinical</b> brain. 

For more information, see the [Publication at Analytical Biochemistry](https://doi.org/10.1016/j.ab.2023.115227) or read the Abstract below.

<hr style="height:5px; visibility:hidden;"/>

## Publication Abstract
Proton (<sup>1</sup>H) Magnetic Resonance Spectroscopy (MRS) is a non-invasive tool capable of quantifying brain metabolite concentrations in vivo. Prioritization of standardization and accessibility in the field has led to the development of universal pulse sequences, methodological consensus recommendations, and the development of open-source analysis software packages. One on-going challenge is methodological validation with ground-truth data. As ground-truths are rarely available for <i>in vivo</i> measurements, data simulations have become an important tool. The diverse literature of metabolite measurements has made it challenging to define ranges to be used within simulations. Especially for the development of deep learning and machine learning algorithms, simulations must be able to produce accurate spectra capturing all the nuances of <i>in vivo</i> data. Therefore, we sought to determine the physiological ranges and relaxation rates of brain metabolites which can be used both in data simulations and as reference estimates. Using the Preferred Reporting Items for Systematic reviews and Meta-Analyses (PRISMA) guidelines, we've identified relevant MRS research articles and created an open-source database containing methods, results, and other article information as a resource. Using this database, expectation values and ranges for metabolite concentrations and T<sub>2</sub> relaxation times are established based upon a meta-analyses of healthy and diseased brains.

<hr style="height:5px; visibility:hidden;"/>

## Citation:
If you use this database, please cite as:
><b>Gudmundson, A. T.</b>, Koo, A., Virovka, A., Amirault, A. L., Soo, M., Cho, J. H., Oeltzschner G., Edden R. A. E., & Stark, C. E. (2023). Meta-analysis and open-source database for in vivo brain Magnetic Resonance spectroscopy in health and disease. Analytical biochemistry, 676, 115227.

<hr style="height:5px; visibility:hidden;"/>

## Keywords
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
