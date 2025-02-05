package com.tt.javadeeplearning.layer;

import java.util.Random;

public class LSTM extends Layer {

    private int inputSize;

    private int hiddenSize;

    private double[][] Wf, Wi, Wc, Wo;
    private double[][] Uf, Ui, Uc, Uo;
    private double[] bf, bi, bc, bo;

    private double[] cellState, hPrev, input;

    private double[] inputGate, forgetGate, outputGate, candidateCellState;

    public LSTM(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        Wf = randomMatrix(hiddenSize, inputSize);
        Wi = randomMatrix(hiddenSize, inputSize);
        Wc = randomMatrix(hiddenSize, inputSize);
        Wo = randomMatrix(hiddenSize, inputSize);
        Uf = randomMatrix(hiddenSize, hiddenSize);
        Ui = randomMatrix(hiddenSize, hiddenSize);
        Uc = randomMatrix(hiddenSize, hiddenSize);
        Uo = randomMatrix(hiddenSize, hiddenSize);
        bf = randomArray(hiddenSize);
        bi = randomArray(hiddenSize);
        bc = randomArray(hiddenSize);
        bo = randomArray(hiddenSize);

        this.cellState = new double[hiddenSize];
        this.hPrev = new double[hiddenSize];
    }

    @Override
    public double[] forwardSingle(double[] input) {
        this.input = input;
        forgetGate = sigmoid(add(matMul(Wf, input), bf));
        inputGate = sigmoid(add(matMul(Wi, input), bi));
        candidateCellState = tanh(add(matMul(Wc, input), bc));
        cellState = elementWiseMultiply(forgetGate, cellState);
        cellState = add(cellState, elementWiseMultiply(inputGate, candidateCellState));
        outputGate = sigmoid(add(matMul(Wo, input), bo));
        hPrev = elementWiseMultiply(outputGate, tanh(cellState));
        return (hPrev);
    }

    @Override
    public double[] backwardSingle(double[] gradient, double learningRate) {
        double[] tanhCellState = tanh(cellState);
        double[] oneMinusTanhSquared = new double[tanhCellState.length];
        for (int i = 0; i < tanhCellState.length; i++) {
            oneMinusTanhSquared[i] = 1 - (tanhCellState[i] * tanhCellState[i]);
        }
        double[] outputGateGrad = elementWiseMultiply(gradient, tanhCellState);
        double[] cellStateGrad = elementWiseMultiply(gradient, elementWiseMultiply(outputGate, oneMinusTanhSquared));
        double[] forgetGateGrad = elementWiseMultiply(cellStateGrad, cellState);
        double[] inputGateGrad = elementWiseMultiply(cellStateGrad, tanhCellState);
        double[] candidateGrad = elementWiseMultiply(cellStateGrad, inputGateGrad);
        Wf = updateWeights(Wf, forgetGateGrad, input, learningRate);
        Wi = updateWeights(Wi, inputGateGrad, input, learningRate);
        Wc = updateWeights(Wc, candidateGrad, input, learningRate);
        Wo = updateWeights(Wo, outputGateGrad, input, learningRate);
        Uf = updateWeights(Uf, forgetGateGrad, hPrev, learningRate);
        Ui = updateWeights(Ui, inputGateGrad, hPrev, learningRate);
        Uc = updateWeights(Uc, candidateGrad, hPrev, learningRate);
        Uo = updateWeights(Uo, outputGateGrad, hPrev, learningRate);
        bf = updateBias(bf, forgetGateGrad, learningRate);
        bi = updateBias(bi, inputGateGrad, learningRate);
        bc = updateBias(bc, candidateGrad, learningRate);
        bo = updateBias(bo, outputGateGrad, learningRate);
        return (gradient);
    }

    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1 / (1 + Math.exp(-x[i]));
        }
        return (result);
    }

    private double[] tanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.tanh(x[i]);
        }
        return (result);
    }

    private double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return (result);
    }

    private double[] elementWiseMultiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return (result);
    }

    private double[][] randomMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        Random random = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextDouble() * 0.01;
            }
        }
        return (matrix);
    }

    private double[] randomArray(int size) {
        double[] array = new double[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            array[i] = random.nextDouble() * 0.01;
        }
        return (array);
    }

    private double[] matMul(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
        return (result);
    }

    private double[][] updateWeights(double[][] weights, double[] gradient, double[] input, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * gradient[i] * input[j];
            }
        }
        return (weights);
    }

    private double[] updateBias(double[] bias, double[] gradient, double learningRate) {
        for (int i = 0; i < bias.length; i++) {
            bias[i] -= learningRate * gradient[i];
        }
        return (bias);
    }

}
