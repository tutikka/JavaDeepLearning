package com.tt.javadeeplearning.layer;

public class Softmax extends Layer {

    @Override
    public double[] forwardSingle(double[] input) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : input) {
            if (value > max) {
                max = value;
            }
        }
        double[] expValues = new double[input.length];
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            expValues[i] = Math.exp(input[i] - max);
            sum += expValues[i];
        }
        output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = expValues[i] / sum;
        }
        return (output);
    }

    @Override
    public double[] backwardSingle(double[] gradient, double learningRate) {
        int n = output.length;
        input = new double[n];
        for (int i = 0; i < n; i++) {
            double gradSum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    gradSum += gradient[j] * output[i] * (1 - output[j]);
                } else {
                    gradSum += gradient[j] * (-output[i] * output[j]);
                }
            }
            input[i] = gradSum;
        }
        return (input);
    }

    @Override
    public String toString() {
        return (String.format("Softmax"));
    }

}
