package com.pt.ml.automatic.differentiation;

/**
 * 多变量 2*(log(x_1))^2 + 3*(log(x_2))^2 + 5*log(x_1) + 6*log(x_2)的最小值
 */
public class Graph2 {
    InputOp input1 = new InputOp();
    InputOp input2 = new InputOp();

    IOpration logOp1 = new LogOp().fromOp(input1);
    IOpration logOp2 = new LogOp().fromOp(input2);
    IOpration powOp1 = new PowOp(2).fromOp(logOp1);
    IOpration powOp2 = new PowOp(2).fromOp(logOp2);
    IOpration mulOp1 = new MultiOp(2).fromOp(powOp1);
    IOpration mulOp2 = new MultiOp(3).fromOp(powOp2);
    IOpration mulOp3 = new MultiOp(5).fromOp(logOp1);
    IOpration mulOp4 = new MultiOp(6).fromOp(logOp2);

    IOpration resultOp = new AddOp(0)
            .fromOp(mulOp1)
            .fromOp(mulOp2)
            .fromOp(mulOp3)
            .fromOp(mulOp4);

    double forward(double x1, double x2) {
        return resultOp.forward(mulOp1.forward(powOp1.forward(logOp1.forward(input1.forward(x1))))
                + mulOp2.forward(powOp2.forward(logOp2.forward(input2.forward(x2))))
                + mulOp3.forward(logOp1.forward(input1.forward(x1)))
                + mulOp4.forward(logOp2.forward(input2.forward(x2))));
    }

    double[] updateParams(double x1, double x2, double learningRate) {
        double logOp1Forward = logOp1.forward(x1);
        double logOp2Forward = logOp2.forward(x2);
        double powOp1Forward = powOp1.forward(logOp1Forward);
        double powOp2Forward = powOp2.forward(logOp2Forward);
        double mulOp1Forward = mulOp1.forward(powOp1Forward);
        double mulOp2Forward = mulOp2.forward(powOp2Forward);
        double mulOp3Forward = mulOp3.forward(logOp1Forward);
        double mulOp4Forward = mulOp4.forward(logOp2Forward);
        //backward
        double dX1 = resultOp.backward(mulOp1Forward) * mulOp1.backward(powOp1Forward) * powOp1.backward(logOp1Forward) * logOp1.backward(x1) +
                resultOp.backward(mulOp3Forward) * mulOp3.backward(logOp1Forward) * logOp1.backward(x1);

        double dX2 = resultOp.backward(mulOp2Forward) * mulOp2.backward(powOp2Forward) * powOp2.backward(logOp2Forward) * logOp2.backward(x2) +
                resultOp.backward(mulOp4Forward) * mulOp4.backward(logOp2Forward) * logOp2.backward(x2);

        return new double[] {x1 - learningRate * dX1, x2 - learningRate * dX2};
    }

    public static void main(String[] args) {
        Graph2 graph2 = new Graph2();
        System.out.println(graph2.forward(1, 1));

        double x1 = 12;
        double x2 = 30;
        for (int i = 0; i < 5000; i++) {
            double[] tmp = graph2.updateParams(x1, x2, 0.01);
            x1 = tmp[0];
            x2 = tmp[1];
            if (i % 100 == 0) {
                System.out.println(graph2.forward(x1, x2));
            }
        }
        System.out.println("x1=" + x1 + "\tx2=" + x2);
    }
}
