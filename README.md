# Directional Detection

This repo stores some of the scripts used for my 2020 NSS work.

Goal: Using an array of 4 stationary NaI bricks, predict the angle of a gamma source.
Summary: As energy will effect the angular response of the array, investigate possible
ways to address it. Gather data sets of Co60, Cs137, and Ir192 data representing
high/med/low energies. Using the total counts in each detector as input features,
predict the angle of each trial with a KNN. Train a KNN on each dataset (single 
isotope), and then a combined isotope dataset. Test those KNNs on subsets of each
dataset to see the impact of training/testing on the same isotope, different isotopes,
and a combined isotope dataset. Develop a simple isotope predictor to see if you can
first flag what isotope the data is, to then use isotope specific trianing data.

Main Files:

NSS_KNN.py - runs the expriments, saves results, makes figs
knn_4det_functions.py - support functions for expirments
dataset_CoCsIr.npy - Combined dataset of 10,000 each Co, Cs, Ir. 
	Columns: (0-9) 10 energy bins, (10-13) total counts in detectors, X, Y, Z, radius, angle, isotope
dataset_Na.npy - Na22 Only dataset 
Ref135_Co/Cs/Ir.npy - Reference tables: Trial for each angle (0-359) at 1, 3 and 5 m
NSS2020_summary.pdf - Sumamary of project submitted to NSS 2020
NSS2020_slides.pdf - Presentation given at NSS 2020 


Legacy Files:
KNN_by_Isotope.py  - Earlier version of expirements
	
