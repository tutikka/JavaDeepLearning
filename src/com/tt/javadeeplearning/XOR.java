package com.tt.javadeeplearning;

import com.tt.javadeeplearning.activation.Sigmoid;
import com.tt.javadeeplearning.activation.TanH;
import com.tt.javadeeplearning.initialization.Random;
import com.tt.javadeeplearning.layer.Activation;
import com.tt.javadeeplearning.layer.Dense;
import com.tt.javadeeplearning.layer.Dropout;
import com.tt.javadeeplearning.loss.BinaryCrossEntropy;
import com.tt.javadeeplearning.metrics.Metrics;
import com.tt.javadeeplearning.network.Network;
import com.tt.javadeeplearning.postprocess.PostProcess;

import java.io.File;

public class XOR {

    public static void main(String[] args) {

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
        network.setTrain(true);
        network.train(trainX, trainY, new BinaryCrossEntropy(), 1000, 0.1);
        network.setTrain(false);

        // save model to file
        network.save(new File("xor.ser"));

        // load model from file
        network = Network.load(new File("xor.ser"));
        System.out.println(network);

        // use train set as test set (mainly for checking that code is ok)
        double[][] testX = trainX;
        double[] testY = trainY;

        // predict label for each set of test features, and calculate accuracy
        int[] predictedY = new int[testY.length];
        for (int i = 0; i < testY.length; i++) {
            predictedY[i] = PostProcess.binaryThreshold(network.predict(testX[i])[0], 0.5);
        }
        System.out.println(String.format("accuracy based on test set of %d items is %s ", testY.length, Metrics.accuracy(testY, predictedY)));
        System.out.println(String.format("binary precision based on test set of %d items is %s ", testY.length, Metrics.binaryPrecision(testY, predictedY)));

    }

}
