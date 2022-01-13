## Task 1
##### For detecting dorsal or palmar a CNN model has been used with 3 hidden layers and an input size of 100x100 pictures.

### Preparing Data
##### To prepare the data loop over all folder with their name stored in an array. Then read all the files into an array of images whose shape will be (-1, 160, 120, 3). 160 and 120 are our picture size and 3 is the number of channels (RGB).

##### This data is also then saved in a .npy file to avoid repeatedly preparing the same data. Upon further use these saved files are loaded instead of preparing the full data again.

### Structure of Model
##### The CNN model has 7 hidden layers. The input layer is a 2d convolution layer using the relu activation function. All conv2d and dense layers are using relu activation function (except the output layer) because it prevents the exponential growth in the computation required making it much more easier and faster to tune your model without compromising on accuracy/perfomance.

##### It has 3 pairs of Conv2d (of 64,256,128) and MaxPooling layers. Which are followed by a Flatten layer flattens the multidimensional input tensors into a single dimension. This is followed by a Dropout(rate=0.5) layer. This layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Followed by a Dense layers (128) and finally an output dense layer using the sigmoid activation function with 1 unit for our 2 categories.

### Compilation and training of model
##### The model uses the binary_crossentropy loss function, which works better for this dataset even though the categories are mutually exclusive.

##### The RMSprop optimizer function is used. RMSprop is gradient based and decreases the step/momentum for large gradients while increasing it for smaller ones. This makes sure model doesn't suffer gradient explosion and/or vanishing

##### The model is then trained with around 5 epochs because of the lack of time.

### Testing
##### The code loops over each image in the test folder and runs the prediction on it. Then stores this data in a csv with pandas library
