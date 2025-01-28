package com.tt.javadeeplearning.postprocess;

public class PostProcess {

    public static int argmax(double[] data) {
        int result = 0;
        double d = data[result];
        for (int i = 1; i < data.length; i++) {
            if (data[i] > d) {
                result = i;
                d = data[result];
            }
        }
        return (result);
    }

    public static int binaryThreshold(double d, double t) {
        return (d > t ? 1 : 0);
    }

}
