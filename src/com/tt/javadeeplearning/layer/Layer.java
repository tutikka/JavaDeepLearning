package com.tt.javadeeplearning.layer;

public abstract class Layer {

    double[] input;

    double[] output;

    public abstract double[] forwardSingle(double[] input);

    public abstract double[] backwardSingle(double[] gradient, double learningRate);

}
