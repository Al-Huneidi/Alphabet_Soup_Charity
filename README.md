# Alphabet_Soup_Charity Project Overview

Determine which organizations should receive donations by predicting the success of a venture paid by Alphabet Soup. Only those projects likely to be a success will receive any future funding from Alphabet Soup. Project consists of preparing input data, creating Deep Learning Models, designing, training, evaluating, and exporting neural networks using Python Pandas, TensorFlow, SciKit Learn.


# Challenge

## Objective

Create a binary classifier that is capable of predicting whether or not an applicant will be successful if funded by Alphabet Soup using the features collected in the provided dataset.

## Challenge Analysis Report

### Inspected the Data
List of features:

	•	EIN and NAME—Identification columns
	•	APPLICATION_TYPE—Alphabet Soup application type
	•	AFFILIATION—Affiliated sector of industry
	•	CLASSIFICATION—Government organization classification
	•	USE_CASE—Use case for funding
	•	ORGANIZATION—Organization type
	•	STATUS—Active status
	•	INCOME_AMT—Income classification
	•	SPECIAL_CONSIDERATIONS—Special consideration for application
	•	ASK_AMT—Funding amount requested
	•	IS_SUCCESSFUL—Was the money used effectively

The target for the model: IS_SUCCESSFUL

The variables that are the features for the model: 

APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT

### Preprocessed the Data

The unnecessary features removed from the input data: Name and EIN

Got the value counts for:  CLASSIFICATION and APPLICATION_TYPE 

Created density plots for:

	- CLASSIFICATION Image
	
	- APPLICATION_TYPE Image
	
Bucketed features: CLASSIFICATION and APPLICATION_TYPE

Encoded categorical variables using one-hot encoding. 

Standardized numerical variables using Scikit-Learn’s StandardScaler class.

### Models Used: Random Forest and Deep Neural Network

I wanted to see the results of each of these models especially since the Random Forest is robust against overfitting and outliers and nonlinear data while being efficient on a large dataset, with less code and faster results.	

#### Random Forest

Random Forest Accuracy: 71% Image

Deep Neural Network
I created a deep neural network to see if it would reach 75% or higher accuracy.
I started with 2 hidden layers as that seemed to make sense with the number of rows in the dataset.  I kept the 2 hidden layers throughout.  In the process adjusted the neurons.
I started with 132 neurons in the first layer and 33 neurons in the second layer.  I chose 132 neurons for the first layer because that number is three times the number of columns in the model dataframe.  I chose 33 neuron for the second layer I arbitrarily chose it by dividing 132 by 4, a fourth of the number of neuron in the first layer.  I set the number of Epochs to 100.  It took a long time to compile and the accuracy was close to the target of higher than 75% at 72.47% with a loss of 55.52%.  
	

### Model Adjustments to Achieve Predictive Accuracy Higher than 75%

Initial model consisted of  the following:

- CLASSIFICATION buckets set at less than 1883 - I selected 1883 after reviewing the value counts for this column and creating the density plot of the values.
- APPLICATION_TYPE buckets set at less than 500 - I selected 500 after reviewing the value counts for this column and creating the density plot of the values.
- 2 hidden layers, first with 132 neurons and the second with 33
- 100 epochs
- reLu activation of first and second hidden layers
- Sigmoid activation for output

Results:
Loss metric: 55.24%	Accuracy: 72.54% 
Image


I had a hypothesis that model was overfitting due to too many epochs. To test my hypothesis of overfitting and to attempt achieving the higher than 75% accuracy and reduce the time to compile I began by adjusting the epochs.
		
1. I reduced the epochs, not increased.
	⁃	80 Epochs: Loss = 55.52%, Accuracy = 72.53%  => an improvement all around
	⁃	65 Epochs: Loss = 55.25%, Accuracy = 72.67%  => improvement in time and accuracy
	⁃	50 Epochs: Loss = 55.25%, Accuracy = 72.74%  => improvement in time and accuracy
  
* It seems more than 50 epochs the model begins to overfit so I stayed with 50 epochs as the optimal number of epochs.

2. I adjusted the bucketing process.
	- I increased the amount in the Other bucket for the APPLICATION_TYPE to include values less than 700.  The loss went up to 55.30% and the accuracy dropped a bit to 72.65%.  
After adjusting the bucket, I noticed the number of parameters dropped from 10,099 from 10,231 as the number of columns in the dataframe dropped by 1.  I interpreted this to mean that a certain number of columns must remain for the accuracy not to drop and for the loss not to increase.  I abandoned adjusting the buckets further.

3. I checked the dataset to see if I could improve the accuracy by adjusting the dataset.
	- I bucketed the ASK_AMT columns to see what would happen to the input data.  When I tried several different amounts to create the “Other” the amounts in the buckets didn’t make any sense as the the values in the ASK_AMT vary too much from each other.  I decided the data was fine.
  
4. I adjusted the number of neurons in each hidden layer.  Since reducing the epochs had a positive effect on the performance and little change to the loss and accuracy I decided to reduce neurons.
	I reduced the number of neurons to 80 in the first layer and 25 in the second layer.
	The loss and accuracy were very close to my initial numbers but the speed of compiling increased significantly so I felt like this was the number of neurons to use for my final model. Loss: 55.17%  Accuracy: 72.46%
		
5. I did not choose a different activation feature for the hidden layers as I thought reLu was the best option for the type of features in this dataset.


Final Deep Learning Neural Network Model

After all the adjustments to the number of neurons, the epochs and the buckets the best results came from a deep neural network model consisting of:
	- CLASSIFICATION buckets set at less than 1883
	- APPLICATION_TYPE buckets set at less than 500
	- 2 hidden layers, the first with 80 neurons and the second with 25
	- 50 epochs
	- reLu activation of first and second hidden layers
	-Sigmoid activation for output

Results:
Final model loss metric: 55.24%	Final model predictive accuracy: 72.54% Image

If I were to try another model, I would try SVM, Support Vector Machine (Classifier) as it is faster to implement, much less code is required to train and test the model.


Resource:

charity_data.csv
