## Task 2
#### For detecting company logos a CNN model has been used with 13 hidden layers and an input size of 100x100 pictures.

### Preparing Data
##### To prepare the data loop over all folder with their name stored in an array. Then read all the files into an array of images whose shape will be (-1, 100, 100, 3). 100 is our picture size and 3 is the number of channels (RGB).

##### This data is also then saved in a .npy file to avoid repeatedly preparing the same data. Upon further use these saved files are loaded instead of preparing the full data again.

### Structure of Model
##### The CNN model has 13 hidden layers. The input layer is a 2d convolution layer using the relu activation function. All conv2d and dense layers are using relu activation function (except the output layer) because it prevents the exponential growth in the computation required making it much more easier and faster to fine tune your model without compromising on accuracy/perfomance.

##### It has 5 pairs of Conv2d (of 64,64,128,128,256) and MaxPooling layers. Which are followed by a Flatten layer. This flattens the multidimensional input tensors into a single dimension. This is followed by a Dropout(rate=0.5) layer. This layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Followed by 2 Dense layers (256, 512) and finally an output dense layer using the softmax activation function with 8 units for our 8 categories.

### Compilation and training of model
##### The model uses the categorical_crossentropy loss function, which works better for this dataset even though the categories are mutually exclusive.

##### The RMSprop optimizer function is used. RMSprop is gradient based and decreases the step/momentum for large gradients while increasing it for smaller ones. This makes sure model doesn't suffer gradient explosion and/or vanishing

##### The model is then trained with around 50 epochs working best. It uses an validation split of 0.2 to make sure the model isn't overfitting

### Testing
##### The code loops over each image in the test folder and runs the prediction on it. Then stores this data in a csv with pandas library
