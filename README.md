# Java Deep Learning

A neural network implementation written in pure Java using no additional libraries. This project is mainly for educational purposes to understand how neural networks work.

Note that batch processing is not implemented, and each "observation" is fed to the network separately, which makes training quite inefficient, but perhaps a little bit easier to understand. 

This project was inspired by:

https://www.youtube.com/@independentcode

## XOR Example

The XOR function is a nice way of testing the implementation, since it is very well known and compact. Additionally, a solution cannot be represented by a linear model, which makes it a suitable task for neural networks.

https://en.wikipedia.org/wiki/XOR_gate

The example below uses all combinations of 0 and 1 as features, labeled with each combination's XOR function output, as the training data. 

**Code**

```java
// initialize features as all combinations of 0 and 1
double[][] trainX = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};

// initialize labels as result of XOR function for each combination
double[] trainY = new double[]{0, 1, 1, 0};

// initialize network with layers
Network network = new Network();
network.addLayer(new Dense(2, 3, new Random())); // each combination has 2 features
network.addLayer(new Activation(new TanH()));
network.addLayer(new Dense(3, 1, new Random())); // binary output (0 or 1)
network.addLayer(new Activation(new Sigmoid()));

// train model
network.train(trainX, trainY, new BinaryCrossEntropy(), 1000, 0.1);

// use train set as test set (mainly for checking that code is ok)
double[][] testX = trainX;
double[] testY = trainY;

// use model to predict XOR result for each combination of 0 and 1 in test set and calculate accuracy
int a = 0;
for (int i = 0; i < testY.length; i++) {
    double t = testY[i];
    // binary output: use probability itself as prediction
    double p = PostProcess.binaryThreshold(network.predict(testX[i])[0], 0.5);
    if (p == t) {
        a++;
    }
}
System.out.print(String.format("accuracy based on test set of %d items is %s ", testY.length, (double) a / (double) testY.length));
```

**Output**

```
epoch 1/1000 | average loss 0.714576
...
epoch 1000/1000 | average loss 0.000000
accuracy based on test set of 4 items is 1.0 
```

## Iris Example

The IRIS dataset is one of the earliest datasets appearing in examples relating to classification. Each observation in the dataset represents an iris flower, including its dimensions (features) and species (label).

https://en.wikipedia.org/wiki/Iris_flower_data_set

Original publication:

https://archive.ics.uci.edu/dataset/53/iris

**Code**

```java
// read features (sepal length and width, petal length and width) using helper
double[][] features = features("data/iris/iris.data");

// read respective labels (setosa, verticolor or virginica)
double[] labels = labels("data/iris/iris.data");

// split into train and test sets
int split = (int) (labels.length * 0.8);

// train features
double[][] trainX = Arrays.stream(features, 0, split).toArray(double[][]::new);

// train labels
double[] trainY = Arrays.stream(labels, 0, split).toArray();

// initialize netowork with layers (first input size represents 4 features, last output 3 labels)
Network network = new Network();
network.addLayer(new Dense(4, 32, new Xavier()));
network.addLayer(new Activation(new ReLu()));
network.addLayer(new Dense(32, 16, new Xavier()));
network.addLayer(new Activation(new ReLu()));
network.addLayer(new Dense(16, 8, new Xavier()));
network.addLayer(new Activation(new ReLu()));
network.addLayer(new Dense(8, 3, new Xavier()));
network.addLayer(new Softmax());

// train model
network.train(trainX, trainY, new CategoricalCrossEntropy(), 100, 0.01);

// save trained model (example if reusing trained model)
network.save(new File("iris.ser"));

// load saved model
network = Network.load(new File("iris.ser"));
System.out.println(network);

// test features and labels based on train/test split
double[][] testX = Arrays.stream(features, split, labels.length).toArray(double[][]::new);
double[] testY = Arrays.stream(labels, split, labels.length).toArray();

// predict label for each set of test features, and calculate accuracy
double[] predictedY = new double[testY.length];
for (int i = 0; i < testY.length; i++) {
    predictedY[i] = PostProcess.argmax(network.predict(testX[i]));
}
System.out.print(String.format("accuracy based on test set of %d items is %s ", testY.length, Metrics.accuracy(testY, predictedY)));
```

**Output**

```
max sepal length 7.900000
max sepal width 4.400000
max petal length 6.900000
max petal width 2.500000
epoch 1/100 | average loss 0.933290
...
epoch 100/100 | average loss 0.143359
Dense (input size = 4, output size = 32, initialization = Xavier)
Activation (function = ReLu)
Dense (input size = 32, output size = 16, initialization = Xavier)
Activation (function = ReLu)
Dense (input size = 16, output size = 8, initialization = Xavier)
Activation (function = ReLu)
Dense (input size = 8, output size = 3, initialization = Xavier)
Softmax
accuracy based on test set of 30 items is 1.0 
```

## MNIST Example

The MNIST dataset represents thousands of handwritten digits (0-9) as 28x28 pixel grayscale images, accompanied by each respective label. The dataset is already divided into separate sets to train the model and predict digits on "new" samples.

http://yann.lecun.com/exdb/mnist/

**Code**

```java
// read image data (features) using helper method to array of images (arrays of 28x28 pixels)
double[][] trainX = images("data/mnist/train-images.idx3-ubyte");

// read labels (0-9) for each respective image
double[] trainY = labels("data/mnist/train-labels.idx1-ubyte");

// display random image (and label) to check if our input is ok
Random random = new Random();
int sampleIndex = random.nextInt(0, trainY.length - 1);
double[] sampleX = trainX[sampleIndex];
double sampleY = trainY[sampleIndex];
System.out.println(String.format("displaying sample from index %d with label %d", sampleIndex, (int) sampleY));
image(sampleX, 28, 28);

// initialize network with layers
Network network = new Network();
network.addLayer(new Dense(784, 256, new Xavier())); // each image has 28x28 = 784 pixels as features
network.addLayer(new Activation(new TanH()));
network.addLayer(new Dense(256, 10, new Xavier())); // categorical ouput; 10 items, each containing probability for respective digit (0-9)
network.addLayer(new Softmax());

// train model
network.train(trainX, trainY, new CategoricalCrossEntropy(), 40, 0.01);

// read new set of images and labels to test with
double[][] testX = images("data/mnist/t10k-images.idx3-ubyte");
double[] testY = labels("data/mnist/t10k-labels.idx1-ubyte");

// use model to predict digit for each image in test set and calculate accuracy
int a = 0;
for (int i = 0; i < testY.length; i++) {
    double t = testY[i];
    // categorical output: use label with highest probability as prediction
    double p = PostProcess.argmax(network.predict(testX[i]));
    if (p == t) {
        a++;
    }
}
System.out.print(String.format("accuracy based on test set of %d items is %s ", testY.length, (double) a / (double) testY.length));
```

**Output**

```
found 60000 images with size 28x28 from the MNIST dataset
found 60000 labels from the MNIST dataset
displaying sample from index 48815 with label 0
                                
               ***          
              *****         
             *******        
             *******        
             *******        
           *********        
           ****   ***       
          ****    ***       
         *****    ***       
         ***      ***       
        ****      ***       
        ***       ***       
       ****       **        
       ***       ***        
      ****     *****        
      ****   ******         
      ************          
      ************          
      **********            
       *******              
                            
epoch 1/40 | average loss 0.979941
...
epoch 40/40 | average loss 0.043388
found 10000 images with size 28x28 from the MNIST dataset
found 10000 labels from the MNIST dataset
accuracy based on test set of 10000 items is 0.9724 
```