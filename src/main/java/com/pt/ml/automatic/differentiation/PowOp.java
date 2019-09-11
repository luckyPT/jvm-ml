package com.pt.ml.automatic.differentiation;

public class PowOp extends AbstarctOp {
    int pow;

    public PowOp(int pow) {
        this.pow = pow;
    }

    @Override
    public double forward(double x) {
        return Math.pow(x, pow);
    }

    @Override
    public double backward(double x) {
        return pow * Math.pow(x, pow - 1);
    }

}
