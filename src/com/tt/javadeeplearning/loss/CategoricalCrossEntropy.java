package com.tt.javadeeplearning.loss;

public class CategoricalCrossEntropy implements Loss {

    @Override
    public double loss(double truth, double[] predicted) {
        int t = (int) truth;
        double p = predicted[t];
        double epsilon = 1e-9;
        p = Math.max(p, epsilon);
        return (-Math.log(p));
    }

    @Override
    public double[] gradient(double truth, double[] predicted) {
        int t = (int) truth;
        double[] gradients = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            if (i == t) {
                gradients[i] = predicted[i] - 1;
            } else {
                gradients[i] = predicted[i];
            }
        }
        return (gradients);
    }
}
