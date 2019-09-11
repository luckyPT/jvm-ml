package com.pt.ml.automatic.differentiation;

/**
 * e为底
 */
public class LogOp extends AbstarctOp {
    @Override
    public double forward(double x) {
        return Math.log(x);
    }

    @Override
    public double backward(double x) {
        return 1 / x;
    }
}
