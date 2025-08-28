Instructions on how to run different experiments:
Dataset needs to be downloaded from https://www.cancerimagingarchive.net/collection/cbis-ddsm and should be placed in ./data/CBIS-DDSM TCIA 
This folder should include a set of csv files and a folder with the name beginning "manifest" which contains the image files. 

For experiments on Calcification dataset use ./BCE/mammogram_research_stratified.ipynb (this is the larger dataset and provided better accuracy on classification).
For experiments on Mass dataset use ./BCE/mammogram_research_mass.ipynb. This has minimal testing as results were worse than those using the calcification 
You will need to replace the following lines with your file paths but it is likely they will be very similar or the same:
dataset_root = "./data/CBIS-DDSM TCIA/manifest-ZkhPvrLo5216730872708713142/"
csv_path_for_train_dataset = './data/CBIS-DDSM TCIA/mass_case_description_train_set.csv'
csv_path_for_test_dataset = './data/CBIS-DDSM TCIA/mass_case_description_test_set.csv'

Modify the CONFIG in the ipynb to change the hyperparameters, parameters, etc.

For cross validation use ./BCE/cross_validation.ipynb

For the paired t-test use ./BCE/paired_t_test.py


