package com.tt.javadeeplearning;

import com.tt.javadeeplearning.activation.ReLu;
import com.tt.javadeeplearning.activation.TanH;
import com.tt.javadeeplearning.initialization.Xavier;
import com.tt.javadeeplearning.layer.Activation;
import com.tt.javadeeplearning.layer.Dense;
import com.tt.javadeeplearning.layer.Softmax;
import com.tt.javadeeplearning.loss.CategoricalCrossEntropy;
import com.tt.javadeeplearning.metrics.Metrics;
import com.tt.javadeeplearning.network.Network;
import com.tt.javadeeplearning.postprocess.PostProcess;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IRIS {

    public static int labelFromString(String label) {
        switch (label) {
            case "Iris-setosa": return (0);
            case "Iris-versicolor": return (1);
            case "Iris-virginica":return (2);
        }
        return (-1);
    }

    public static String labelFromInt(int i) {
        switch (i) {
            case 0: return ("Iris-setosa");
            case 1: return ("Iris-versicolor");
            case 2: return ("Iris-virginica");
        }
        return (null);
    }

    public static double[] labels(String path) {
        try {
            List<Double> l = new ArrayList<>();
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                String label = line.split(",")[4];
                l.add((double) labelFromString(label));
            }
            return (l.stream().mapToDouble(d -> d).toArray());
        } catch (Exception e) {
            return (null);
        }
    }

    public static double[][] features(String path) {
        try {
            List<double[]> l = new ArrayList<>();
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            double maxSepalLength = 0;
            double maxSepalWidth = 0;
            double maxPetalLength = 0;
            double maxPetalWidth = 0;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                double sepalLength = Double.parseDouble(line.split(",")[0]);
                if (sepalLength > maxSepalLength) {
                    maxSepalLength = sepalLength;
                }
                double sepalWidth = Double.parseDouble(line.split(",")[1]);
                if (sepalWidth > maxSepalWidth) {
                    maxSepalWidth = sepalWidth;
                }
                double petalLength = Double.parseDouble(line.split(",")[2]);
                if (petalLength > maxPetalLength) {
                    maxPetalLength = petalLength;
                }
                double petalWidth = Double.parseDouble(line.split(",")[3]);
                if (petalWidth > maxPetalWidth) {
                    maxPetalWidth = petalWidth;
                }
                l.add(new double[]{sepalLength, sepalWidth, petalLength, petalWidth});
            }
            System.out.println(String.format("max sepal length %f", maxSepalLength));
            System.out.println(String.format("max sepal width %f", maxSepalWidth));
            System.out.println(String.format("max petal length %f", maxPetalLength));
            System.out.println(String.format("max petal width %f", maxPetalWidth));
            double[][] d = new double[l.size()][];
            for (int i = 0; i < d.length; i++) {
                double[] f = l.get(i);
                f[0] /= maxSepalLength;
                f[1] /= maxSepalWidth;
                f[2] /= maxPetalLength;
                f[3] /= maxPetalWidth;
                d[i] = f;
            }
            return (d);
        } catch (Exception e) {
            return (null);
        }
    }

    public static void main(String[] args) {

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
    }

}
