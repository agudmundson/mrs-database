# mrs-database:
 Open-source database for in vivo brain MRS results.

## Citation:
>Gudmundson, A. T., Koo, A., Virovka, A., Amirault, A. L., Soo, M., Cho, J. H., ... & Stark, C. E. (2023). Meta-analysis and open-source database for in vivo brain Magnetic Resonance spectroscopy in health and disease. Analytical biochemistry, 676, 115227.

## Link:
[Link to Publication at Analytical Biochemistry](https://doi.org/10.1016/j.ab.2023.115227)

## Summary:
This Repository hosts an open-source database for in vivo 1H Magnetic Resonance Spectroscopy (MRS) of the brain. Users can download, use, and contribute their own results to this database.

This database was originally created for a ***Meta-analysis*** investigating the standard ranges of metabolite concentrations and relaxation times in the ***Healthy*** and different ***Clinical*** brain. For more information, see the Abstract below:

>Proton (1H) Magnetic Resonance Spectroscopy (MRS) is a non-invasive tool capable of quantifying brain metabolite concentrations in vivo. Prioritization of standardization and accessibility in the field has led to the development of universal pulse sequences, methodological consensus recommendations, and the development of open-source analysis software packages. One on-going challenge is methodological validation with ground-truth data. As ground-truths are rarely available for in vivo measurements, data simulations have become an important tool. The diverse literature of metabolite measurements has made it challenging to define ranges to be used within simulations. Especially for the development of deep learning and machine learning algorithms, simulations must be able to produce accurate spectra capturing all the nuances of in vivo data. Therefore, we sought to determine the physiological ranges and relaxation rates of brain metabolites which can be used both in data simulations and as reference estimates. Using the Preferred Reporting Items for Systematic reviews and Meta-Analyses (PRISMA) guidelines, we've identified relevant MRS research articles and created an open-source database containing methods, results, and other article information as a resource. Using this database, expectation values and ranges for metabolite concentrations and T2 relaxation times are established based upon a meta-analyses of healthy and diseased brains.

Inside you'll find:
1. Analysis_00_Confirm_Supp_Fx.ipynb 
  - A Jupyter notebook that demonstrates the Supplemental Meta-analysis tools used within the analyses.

2. Analysis_01_Healthy_Groups.ipynb 
  - A Jupyter notebook with the brain metabolite concentration analysis in ***Healthy*** Populations.

3. Analysis_02_Control_Groups.ipynb 
  - A Jupyter notebook with the brain metabolite concentration analysis in ***Clinical*** Populations.

4. Analysis_03_T2_Groups.ipynb 
  - A Jupyter notebook with the brain metabolite relaxation analysis. Here, a meta-regression was developed that can predict T2 relaxation given the experimental conditions.

## Keywords
Human brain, Database, Meta-analysis, Proton MRS, In vivo
