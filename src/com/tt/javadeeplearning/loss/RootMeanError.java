package com.tt.javadeeplearning.loss;

public class RootMeanError implements Loss {

    @Override
    public double loss(double truth, double[] predicted) {
        double result = 0;
        for (int i = 0; i < predicted.length; i++) {
            result += Math.pow(truth - predicted[i], 2);
        }
        result /= predicted.length;
        return (result);
    }

    @Override
    public double[] gradient(double truth, double[] predicted) {
        double[] result = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            result[i] = (2 * (predicted[i] - truth)) / predicted.length;
        }
        return (result);
    }

}
