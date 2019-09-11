package com.pt.ml.automatic.differentiation;

public class AddOp extends AbstarctOp {
    double bias;

    public AddOp(double bias) {
        this.bias = bias;
    }

    @Override
    public double forward(double x) {
        return x + bias;
    }

    @Override
    public double backward(double x) {
        return 1;
    }
}
