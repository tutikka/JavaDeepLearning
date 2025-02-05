package com.tt.javadeeplearning.loss;

public class SparseCategoricalCrossEntropy implements Loss {

    @Override
    public double loss(double truth, double[] predicted) {
        return (-Math.log(predicted[(int) truth]));
    }

    @Override
    public double[] gradient(double truth, double[] predicted) {
        double[] gradient = new double[predicted.length];
        gradient[(int) truth] = -1.0 / predicted[(int) truth];
        return (gradient);
    }

}
