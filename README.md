# Java Deep Learning

A neural network implementation written in pure Java using no additional libraries. This project is mainly for educational purposes to understand how neural networks work.

## XOR Example

```java
// initialize features as all combinations of 0 and 1
double[][] trainX = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};

// initialize labels as result of XOR function for each combination
double[] trainY = new double[]{0, 1, 1, 0};

// initialize network with layers
Network network = new Network();
network.addLayer(new Dense(2, 3, new Random()));
network.addLayer(new Activation(new TanH()));
network.addLayer(new Dense(3, 1, new Random()));
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

## MNSIT Example

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
network.addLayer(new Dense(784, 256, new Xavier()));
network.addLayer(new Activation(new TanH()));
network.addLayer(new Dense(256, 128, new Xavier()));
network.addLayer(new Activation(new TanH()));
network.addLayer(new Dense(128, 10, new Xavier()));
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
