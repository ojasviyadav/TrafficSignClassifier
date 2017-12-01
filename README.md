# TrafficSignClassifier
• Built and trained a deep neural network to classify German traffic signs • Used TensorFlow to build the neural net. Used Normalization and MaxPooling to guard against overfitting • 95% testing accuracy

1. About the Data set
                I used the standard python functions and NumPy library to calculate summary statistics of the traffic signs data set:
                Number of training examples = 34799
                Number of testing examples = 12630
                Number of validation examples = 4410
                Image data shape = (34799, 32, 32, 3)
                Number of classes = 43
           The DataSet is also visualized in the .ipynb file

2. Pre-processing : 
                1.Cropped the image from 32x32 to 30x30 as most of the pictures had extra space on the edges. This increased the processing speed and removed areas which could confuse the network.
                2.Normalised the data by : Dividing all the pixels by 255
                Subtracting the result by 0.5. This made sure the mean of all the data was zero and the unit standard deviation.
                3.I also experimented with rotation however the accuracy always happened to be lesser than when not rotation wasn’t employed. 

3. Model Architecture : 

        My final model consisted of the following layers:

        Layers used : 6

        1.	Layer 1 had two parallel convolutions which concatenate. These are the parallel layers :
        1.	  Convolutional layer. 
        Input : 30x30x3
                Output : 24x24x5
                Padding : VALID
                Dropout applied

        2.	  Convolutional Layer
        Input : 30x30x3
        Output : 22x22x7
        Padding : VALID
        Dropout applied
        General intuition behind two parallel and different layers is that the different filter sizes will capture different details of the input image.
        Similarly the number of output layers of both the layers are also different.
        The second layer gets to compensate for its lower output size by getting higher number of output layers.

        These two layers are resized to 23x23 so both the layers have equal dimensions before getting concatenated. These two are concatenated to form a batch of size 23x23x12. This is passed through a relu function.

        2.	Layer 2 is just a 1x1 convolutional layer to make the network deeper
        3.	Layer 3 is again forks into 2 parallel layers with different filer sizes to capture different sorts of patterns
        1.	Convolutional layer
        Input : 23x23x12
        Output : 16x16x25
        Padding : VALID
        Dropout applied

        2.	Convolutional layer
        Input : 23x23x12
        Output : 18x18x23
        Padding : VALID
        Dropout applied
        Again, first layer compensates for its lesser output size by having more number of output layers. Both the layers are resized to 17x17 to ready them for concatenation. After concatenation the result is passed through a relu function having dimensions 17x17x48.
        This is MaxPooled to 8x8x48 by filter size 3x3 and stride 2x2
        This is flattened to 3072
            4. A fully connected layer. Input = 3072. Output = 200. Output is Relu’d.
            5. A fully connected layer. Input = 200. Output = 84. Output is Relu’d.
            6. A fully connected layer. Input = 84. Output = 43. Output is returned.
            
3. How was it trained : 
        The CNN was trained with the Adam optimizer, initial learning rate was 0.001 with 10 percent decrease with each epoch

        batch size = 32 images

        The model was trained for 15 epochs with one dataset.
        Variables were initialized with using of a truncated normal distribution with mu = 0.0 and sigma = 0.1. 
        Dropout of 0.75 was used for training
        Training was performed on a Intel i7 CPU and took about an hour.

4. More information about how I went on about the project

        My final model results were:
        * training set accuracy of 0.963
        * validation set accuracy of 0.932
        * test set accuracy of 0.943

        If an iterative approach was chosen:
                * First I tried the LeNet architecture as I was familiar with it
                * LeNet couldn’t go beyond an accuracy of 0.89, underfitting. My intuition is that it wasn’t deep enough and it couldn’t capture the patterns in one input diversely enough.
                •	*I preserved the last three Fully connected layers because they didn’t seem much at fault. The problem of not being able to observe different patterns diversely was solved by introducing parallel layers in the front and then combining their outputs.
                •	I also applied dropout of 0.9 individually to these parallel layers to prevent overfitting.
                •	Dropout rate was initially 0.5 but the results kept improving up to a rate of 0.75.
                •	After having two sets of parallel layers the output happened to be quite large which slowed down the computation. Hence I applied a MaxPooling layer to significantly reduce the data size. I didn’t use average pooling here because it felt like a lot of pixels in the 17x17 would be empty and averagePooling them would just make things worse as instead of leaving these empty pixels out we would be letting out output be affected by them. 
                •	During researching I found that one shouldn’t apply Pooling and dropout on the same layer. I made sure that didn’t happen with my network.
                •	I also experimented with using elu function instead of relu after reading several sources about its improved results. But elu didn’t perform as good as relu did for me.
                •	Batch Size of 32 improved results considerably even though it slowed the computation
                •	15 epochs seemed just right as beyond 15 the results didn’t improve considerably.

        I chose an architecture with big layers in its beginning because I wanted to capture as many different patterns as possible.

5. How can we further enhance this : 

        1. We can further visualize the layers and observe which parts are redundant and which parts need more resources
        2. We can enhance the pre-processing by rotating, warping, blocking some minor extra part of the view like blocking the circular shape as well as the circle and just training the model based on the '30' written in the sign. This way we make it learn what are the absolute critical parts
        3. We can apply regularization to reduce overfitting
        4. We can apply contrast and exposure transforms to enhance the quality of the images
        
