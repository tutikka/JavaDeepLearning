package com.tt.javadeeplearning;

import com.tt.javadeeplearning.activation.Sigmoid;
import com.tt.javadeeplearning.activation.TanH;
import com.tt.javadeeplearning.initialization.Random;
import com.tt.javadeeplearning.layer.Activation;
import com.tt.javadeeplearning.layer.Dense;
import com.tt.javadeeplearning.loss.BinaryCrossEntropy;
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
        network.train(trainX, trainY, new BinaryCrossEntropy(), 1000, 0.1);

        // save model to file
        network.save(new File("xor.ser"));

        // load model from file
        network = Network.load(new File("xor.ser"));
        System.out.println(network);

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

    }

}
