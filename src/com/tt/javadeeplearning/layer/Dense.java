package com.tt.javadeeplearning.layer;

import com.tt.javadeeplearning.initialization.Initialization;

import java.util.Random;

public class Dense extends Layer {

    private int inputSize = -1;

    private int outputSize = -1;

    private Initialization initialization;

    protected double[][] weights;

    protected double[] bias;

    /**
     * Constructor. Uses random initialization.
     *
     * @param inputSize The number of input neurons
     * @param outputSize The number of output neurons
     */
    public Dense(int inputSize, int outputSize) {
        initialization = new com.tt.javadeeplearning.initialization.Random();
        init(inputSize, outputSize);
    }

    /**
     * Constructor.
     *
     * @param inputSize The number of input neurons
     * @param outputSize The number of output neurons
     * @param initialization The initialization function
     */
    public Dense(int inputSize, int outputSize, Initialization initialization) {
        this.initialization = initialization;
        init(inputSize, outputSize);
    }

    private void init(int inputSize, int outputSize) {
        Random random = new Random();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        weights = new double[inputSize][outputSize];
        bias = new double[outputSize];
        for (int y = 0; y < outputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                weights[x][y] = initialization.initialize(inputSize, outputSize);
            }
            // bias[y] = random.nextDouble();
            bias[y] = 0.01;
        }
    }

    @Override
    public double[] forwardSingle(double[] input) {
        this.input = input;
        output = new double[outputSize];
        for (int y = 0; y < outputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                output[y] += input[x] * weights[x][y];
            }
            output[y] += bias[y];
        }
        return (output);
    }

    @Override
    public double[] backwardSingle(double[] gradient, double learningRate) {
        double[] result = new double[inputSize];
        for (int y = 0; y < outputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                result[x] += weights[x][y] * gradient[y];
                weights[x][y] -= input[x] * gradient[y] * learningRate;
            }
            bias[y] -= gradient[y] * learningRate;
        }
        return (result);
    }

    @Override
    public String toString() {
        return (String.format("Dense (input size = %d, output size = %d, initialization = %s)", inputSize, outputSize, initialization));
    }

}
