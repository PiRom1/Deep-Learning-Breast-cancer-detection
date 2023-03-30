# Deep-Learning-Breast-cancer-detection
My deep learning models to detect breast cancer from mammography.
Every models are coded in Python, with the library TensorFlow. 

Different models are uploaded : 
1) CNN_mammos_simple : A simple CNN which evaluates the presene or absence of cancer on a single mammogram. 
2) CNN_mammos_TimeDistributedWrapper : A CNN followed by a RNN (LSTM), to detect breast cancer on series of 5 mammograms. 
3) CNN_mammos_TimeDistributedWrapper_heterogene : A CNN followed by a RNN (LSTM), to detect breast cancer on heterogeneous series mammograms (from 1 to 5).

You will need to change the path toward your current directory, and download the scrip my_functions.py" which contains hand-made functions to process the data and plot some results. 
