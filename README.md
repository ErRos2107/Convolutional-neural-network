# Convolutional networks
This project apply convolutional neural networks(CNN) with dropout and batch normalization to [EMNIST(Extended MNIST)](https://www.nist.gov/itl/iad/image-group/emnist-dataset) classification problem.

The EMNIST Balanced dataset, with 131,600 characters. 47 balanced classes. EMNIST extends MNIST by including images of handwritten letters (upper and lower case) as well as handwritten digits. There are 47 different labels. Here for the experiments, we randomly chose 100,000 example for training, 15,800examples for validation and 15,800examples for testing.
## CNN layer and baseline
Architecture:
- Conv -> relu
- (Conv -> Relu -> Pool)xN -> Affin -> Relu
- (Conv -> Relu)xN -> Pool -> Affin -> Relu
![](/images/CNN_accuracy_baseline.png)

## CNN with batch normalization
To see whether inserting batch normalization layer before or after the activation layer, we compare two simple model based on base line DNN model and insert a batch normalization layer before and after the relu layer. The hyperparameter is set to be the same as the base line model.
So we will try two experiments:
(Conv -> RELU -> Pool)x2 -> BatchN -> RELU
(Conv -> RELU -> Pool)x2 -> RELU -> BatchN
![](/images/CNN_accuracy_BNre.png)
![](/images/CNN_error_BNre.png)

## more...
