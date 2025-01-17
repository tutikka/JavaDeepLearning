package com.tt.javadeeplearning.initialization;

public class Random implements Initialization {

    private java.util.Random random = new java.util.Random();

    @Override
    public double initialize(int inputSize, int outputSize) {
        return (random.nextDouble());
    }

}
