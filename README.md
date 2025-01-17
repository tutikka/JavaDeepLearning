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
network.addLayer(new Dense(256, 128, new Xavier()));
network.addLayer(new Activation(new TanH()));
network.addLayer(new Dense(128, 10, new Xavier())); // categorical ouput; 10 items, each containing probability for respective digit (0-9)
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
displaying sample from index 20210 with label 5
                                        
              ********      
           ************     
           ************     
          ****              
          ****              
          ****              
          ***               
          ***               
          ****              
          *****             
          *****             
           ****             
            ****            
            *****           
             ****           
              ***           
             ****           
            *****           
           ******           
            ***             
                                
epoch 1/40 | average loss 2.335045
...
found 10000 images with size 28x28 from the MNIST dataset
found 10000 labels from the MNIST dataset
accuracy based on test set of 10000 items is 0.0958 
```