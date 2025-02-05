package com.tt.javadeeplearning.layer;

import java.util.Random;

public class Dropout extends Layer {

    private double rate;

    private boolean train = true;

    private double[] mask;

    private final Random random = new Random();

    public Dropout(double rate) {
        this.rate = rate;
    }

    public void setTrain(boolean train) {
        this.train = train;
    }

    @Override
    public double[] forwardSingle(double[] input) {
        if (train) {
            mask = new double[input.length];
            for (int i = 0; i < input.length; i++) {
                mask[i] = (random.nextDouble() > rate) ? 1.0 : 0.0;
                input[i] *= mask[i];
            }
        }
        return (input);
    }

    @Override
    public double[] backwardSingle(double[] gradient, double learningRate) {
        if (train) {
            for (int i = 0; i < gradient.length; i++) {
                gradient[i] *= mask[i];
            }
        }
        return gradient;
    }

    @Override
    public String toString() {
        return (String.format("Dropout (rate = %s)", rate));
    }

}
