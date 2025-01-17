package com.tt.javadeeplearning.layer;

import com.tt.javadeeplearning.activation.ReLu;
import com.tt.javadeeplearning.activation.TanH;

public class Activation extends Layer {

    private com.tt.javadeeplearning.activation.Activation activation;

    private int inputSize = -1;

    private int outputSize = -1;

    public Activation() {
        this.activation = new TanH();
    }

    public Activation(com.tt.javadeeplearning.activation.Activation activation) {
        this.activation = activation;
    }

    @Override
    public double[] forwardSingle(double[] input) {
        inputSize = input.length;
        outputSize = inputSize;
        output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            output[i] = activation.activate(input[i]);
        }
        return (output);
    }

    @Override
    public double[] backwardSingle(double[] gradient, double learningRate) {
        inputSize = output.length;
        double[] result = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            result[i] = gradient[i] * activation.gradient(output[i]);
        }
        return (result);
    }

}
