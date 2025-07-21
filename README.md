# ERT

Currently (training model on LTD05.nc only), most of my code is in google colab, as its environment makes it much faster for me to manipulate and visualize the .nc files using xarray. All of my documentation and thoughts will be there as well, at least until enough work is done where basic, step-by-step documentation won't be enough/we get closer to a finished product where overall documentation is necessary. Also included in the repo is the .py file that I use to actually train and run the model, as running in Colab causes it to crash after just 2 epochs. However, it's pretty barebones (just copy paste), to quickly try on my personal machine. The results and my thoughts on it are also all in the Colab. 

https://colab.research.google.com/drive/1MwmT3HSqhgtpULKR72tfgj3o8BhjTznK?usp=sharing (comments allowed) 


## 7/21 Update
used MI/FI scores to do feature selection, selecting categories that had an FI score of above .06  
still having trouble using imbalanced dataset with Keras. experimenting with undersampling and upweighting OR undersampling and ensembling multiple models   
also experimented with using scikit-learn's RandomForestClassifier isntead of Keras, which instantly resulted in much better accuracy, precision, and recall  (.877 macro average, .95 weighted)   
- will keep experimenting witH Keras, but as of now sklearn seems like a much more optimistic route. 
