package com.tt.javadeeplearning;

import com.tt.javadeeplearning.activation.TanH;
import com.tt.javadeeplearning.initialization.Xavier;
import com.tt.javadeeplearning.layer.Activation;
import com.tt.javadeeplearning.layer.Dense;
import com.tt.javadeeplearning.layer.Softmax;
import com.tt.javadeeplearning.loss.CategoricalCrossEntropy;
import com.tt.javadeeplearning.network.Network;
import com.tt.javadeeplearning.postprocess.PostProcess;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.Random;

public class MNIST {

    public static double[] labels(String path) {
        try {
            DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(new File(path))));
            int magic = in.readInt();
            int items = in.readInt();
            double[] result = new double[items];
            for (int i = 0; i < items; i++) {
                result[i] = in.readUnsignedByte();
            }
            in.close();
            System.out.println(String.format("found %d labels from the MNIST dataset", items));
            return (result);
        } catch (Exception e) {
            return (null);
        }
    }

    public static double[][] images(String path) {
        try {
            DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(new File(path))));
            int magic = in.readInt();
            int items = in.readInt();
            int rows = in.readInt();
            int columns = in.readInt();
            double[][] result = new double[items][];
            for (int i = 0; i < items; i++) {
                double[] d = new double[rows * columns];
                int j = 0;
                for (int x = 0; x < rows; x++) {
                    for (int y = 0; y < columns; y++) {
                        d[j] = in.readUnsignedByte() / 255.0;
                        j++;
                    }
                }
                result[i] = d;
            }
            in.close();
            System.out.println(String.format("found %d images with size %dx%d from the MNIST dataset", items, rows, columns));
            return (result);
        } catch (Exception e) {
            return (null);
        }
    }

    public static void image(double[] data, int rows, int columns) {
        int i = 0;
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < columns; y++) {
                double d = data[i];
                System.out.print(d < 0.5 ? " " : "*");
                i++;
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {

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

    }

}
