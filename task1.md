## Task 1
##### For detecting dorsal or palmar a CNN model has been used with 3 hidden layers and an input size of 100x100 pictures.

### Preparing Data
##### To prepare the data loop over all folder with their name stored in an array. Then read all the files into an array of images whose shape will be (-1, 160, 120, 3). 160 and 120 are our picture size and 3 is the number of channels (RGB).

##### This data is also then saved in a .npy file to avoid repeatedly preparing the same data. Upon further use these saved files are loaded instead of preparing the full data again.

### Structure of Model
##### The CNN model has 3 hidden layers. The input layer is a 2d convolution layer using the relu activation function.

##### After that is a MaxPooling2D layer. Which is followed by a Flatten layer. This flattens the multidimensional input tensors into a single dimension. Followed by 1 Dense layer (128) and finally an output dense layer using the sigmoid activation function with 1 units for the probable category.

### Compilation and training of model
##### The model uses the binary_crossentropy loss function, because there are only 2 categories and it works better for this dataset.

##### The SGD optimizer is used with a learning rate of 1e-4 and momentum of 0.9

##### The model is then trained with around 10 epochs working best. It uses an validation split of 0.3 to make sure the model isn't overfitting

### Testing
##### The code loops over each image in the test folder and runs the prediction on it. Then stores this data in a csv with pandas library
