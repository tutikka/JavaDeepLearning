package com.tt.javadeeplearning.initialization;

import java.util.Random;

public class Xavier implements Initialization {

    private Random random = new Random();

    @Override
    public double initialize(int inputSize, int outputSize) {
        return (random.nextDouble() * Math.sqrt(2.0 / (inputSize + outputSize)));
    }

}
