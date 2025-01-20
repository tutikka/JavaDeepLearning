package com.tt.javadeeplearning.initialization;

import java.io.Serializable;

public class Random implements Initialization, Serializable {

    private java.util.Random random = new java.util.Random();

    @Override
    public double initialize(int inputSize, int outputSize) {
        return (random.nextDouble());
    }

    @Override
    public String toString() {
        return ("Random");
    }

}
