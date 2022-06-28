In this repository, we provide 1 .ipynb file and 10 .py files:
	
	For demo :
		- SM_EG.ipynb : notebook file that illustrates all the result of the project.
	
	For data preparation : 
		- create_tiles.py : contains functions to extract tiles from WSIs.
		- concatenate_tiles.py : contains a function to concatenate extracted tiles.
		- stainNorm_Vahadane.py : contains a class of stain normalizer Vahandane method.
		- stain_utils.py :  contains functions useful for stainNorm_Vahadane.py.

	Utility functions:
		- utils.py : contains functions for visualizations/loss functions/reading data.

	For modeling :
		- resnet18.py : contains a class defining the ResNet18 architecture.
		- pooling_resnet18.py : contains a class defining the Pooling ResNet18 architecture.
		- MIL_resnet18.py : contains a class defining the MIL ResNet18 architecture.
	
	For training :
		- train.py : contains functions to train the three previous models. 

	For testing : 
		- test.py : contains functions to test the three previous models.

NB. 
- To run the notebook, please put all the files in the same repository. 
 
- We provide in the following drive link all the data including the pre-processed images we created with our approach. You can use it directly to run the notebook by changing the "Working_dir" variable to the data folder (DLMI_Mhadhbi_Ghamgui).

https://drive.google.com/drive/folders/1GZ8cKy2oUIlJoEvtRpXXaTnHVNbhy2J4?usp=sharing

- We provided in the notebook all code to recreate the pre-processed images. Thus, to do so, please uncomment the commented cells. 

