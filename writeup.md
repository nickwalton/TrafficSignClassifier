#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

[//]: # (Image References)

[image1]: ./histogram.png "Visualization"
[image4]: ./germansignssmall/sign1.png "Traffic Sign 1"
[image5]: ./germansignssmall/sign2.png "Traffic Sign 2"
[image6]: ./germansignssmall/sign3.png "Traffic Sign 3"
[image7]: ./germansignssmall/sign4.png "Traffic Sign 4"
[image8]: ./germansignssmall/sign5.png "Traffic Sign 5"

---
### Writeup / README

#### 1. Project Code

Here is a link to my project code: https://github.com/nickwalton/traffic_sign_classifier

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to get the summary of the data set

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

This is a histogram showing the number of occurences of each classification. All three sets of data are plotted here. But the validation and test set have the same number of samples and appear on top of each other with the same distribution. From this we can see that there is a huge difference in the number of samples for some classifications making certain signs at a higher danger of being misclassified. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first I tried both normalizing and applying a grayscale to the images, however I didn't see any serious improvement from the normaliziation so in the end I only applied grayscale. Grayscale did however make a significant difference on the validation accuracy.

Later I added methods to increase the size of the data set. I first found the image types that had half as many samples as that with the most and I added slightly translated and rotated images to this set until the distribution was more even. 
Then to that set I quadrupled the size of the data set by again randomly rotating between -10 and 10 degrees, randomly translating by a few pixels and increasing and decreasing the brightness

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x1 Grayscale image   				            | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					        |											                        	|
| Max pooling	         	| 2x2 stride,  outputs 14x14x16 				        |
| Dropout				        |											                        	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU				        	|												                        |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x24   	        			|
| Dropout				        |										                        		|
| Flatten			        	| Output = 600								                	|
| Connected		      		| Output = 240					                				|
| RELU			         		|											                        	|
| Dropout				        |										                        		|
| Connected			      	| Output = 120							                		|
| RELU			           	|						  				                         	|
| Output Layer		    	| Output = 43							                  		|
  

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Honestly I spent about half of the time on this project trying to get GPU support to work. Once I did do that I noticed the training time cut by a fourth or so which was very convenient. From playing around with the hyperparameters and comparing the performance I eventually settled on the following hyperparameters.

LEARNING_RATE = 0.0002
EPOCHS = 30
BATCH_SIZE = 128
KEEP_PROB = 0.6

These seemed to provide a satisfactory result and my validation reached up to 0.975

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.976
* test set accuracy of 0.96

I initially started out with the same architecture that we had used in the LeNet lab and decided to build off that. It was simaltaneously more convenient than starting from scratch and was also built for a similar problem. The main architecture changes that I implemented were including dropout which significantly increased the max validation I was able to achieve by I assume helping my network not to overfit or rely on a single feature too much. I also increased the number of nodes at all the layers in order to complensate for the fact that the LeNet we used was built to classify a smaller number of features, (only 10) as opposed to 43 so my intution told me that in order to be effective to do that the network would have to be enabled to have more information passing through it. 

In my iterative process I did several things along the way to slowly increase accuracy, I recorded each change I made and the effect it had on the last five training and validation accuracies to be able to compare different changes and see thier affect. I changed the dropout rate, the learning rate, and some of the filter depths and and the number fully ndoes in fully 
One of the parameters I adjusted that gave me significant boosts was decreasing the sigma for my weight initializion. I'm not exactly sure why this helped my network train faster, but it seems initializing the weights closer to 0 on average greatly helped it. 

After doing these I went back to preprocessing and augmented my data eventually getting around 10x as much data to process. This made a big difference and is the main reason I was able to increase my validation accuracy from around 0.945 to 0.976. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Initally my accuracy to these new signs was 100%, but as I considered it I realized that the images that came up on google image search were by far much clearer than what one might find on the actual road. So I decided to replace image 3 and image 5 with more difficult images to classify. This turned out to be a challenge for my classifier and it misclassified both of them. 

The first image is quite clear, but the captioning by the photography company could potentially limit it. However it was classified just fine. 
The second was also quite clear, though at a slight angle. 
The third was a seriously difficult one. It is more than half obscured and takes up less than a fourth of the image. With the limited in formation in a only 32x32 pixel image this was very difficult for my classifier and it failed. 
The fourth had similar problems to the first, but it was quite clear and didn't fail.
The fifth wasn't quite as difficult as the third, but it has dirt on it, extra lettering as well as a company caption making it a challenge for my classifier as well. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			            |     Prediction	              		| 
|:---------------------:|:---------------------------------:| 
| Slippery Road       	| Slippery Road  							    	| 
| Stop     			        | Stop										          |
| 30 km/h			        	| Go Straight or Right							|
| Yield	      		      | Yield					 		        		    |
| Road Work			        | No Passing for Vehicles      		|


The model was able to correctly guess 5 of the 5 traffic signs initially, but after adding more difficult signs it only guessed 3 out of 5 giving it 60% instead of 100%. This shows that in testing a large amount of data including more difficult to guess data must be added to get an accurate idea of how well our classifier is performing. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------:| 
| 1.00        		    	| Slippery Road   						  		| 
| 0.999     			      | Stop							        				|
| 0.757					        | Go Straight or Right							|
| 1.00	      			    | Yield						 		             	|
| 0.903				          | No Passing for Vehicles      			|


For the three images that it got correct the model was almost certain of it's accuracy. However for the two it got wrong it was surprisingly confident as well with 0.75 and 0.9 confidence levels showing it wasn't well trained for those type of difficult scenarios. 

