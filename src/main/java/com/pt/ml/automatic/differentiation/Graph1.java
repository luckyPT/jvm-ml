package com.pt.ml.automatic.differentiation;

/**
 * 求3*x^2 + 3*x + 5的最小值对应的x
 */
public class Graph1 {
    InputOp input = new InputOp();

    IOpration powOp = new PowOp(2)
            .fromOp(input);
    IOpration mulOp1 = new MultiOp(3).fromOp(input);
    IOpration mulOp2 = new MultiOp(3).fromOp(powOp);
    IOpration resultOp = new AddOp(5)
            .fromOp(mulOp1)
            .fromOp(mulOp2);

    double forward(double x) {
        return resultOp.forward(mulOp2.forward(powOp.forward(input.forward(x))) +
                mulOp1.forward(input.forward(x)));
    }

    double updateParams(double x, double learningRate) {
        double inputForward = input.forward(x);
        double powOpForward = powOp.forward(inputForward);
        double mulOp1Forward = mulOp1.forward(inputForward);
        double mulOp2Forward = mulOp2.forward(powOpForward);
        //backward
        double mulOp2Backward = resultOp.backward(mulOp1Forward + mulOp2Forward) * mulOp2.backward(powOpForward);
        double mulOp1Backward = resultOp.backward(mulOp1Forward + mulOp2Forward) * mulOp1.backward(inputForward);
        double powOpBackWard = mulOp2Backward * powOp.backward(inputForward);
        double dInput = powOpBackWard + mulOp1Backward;
        return x - learningRate * dInput;
    }

    public static void main(String[] args) {
        Graph1 myGraph = new Graph1();
        double initX = 2;
        System.out.println(myGraph.forward(initX));//前向传播
        for (int i = 0; i < 2000; i++) {
            initX = myGraph.updateParams(initX, 0.001);
            if (i % 100 == 0) {
                System.out.println("iter:" + i + ":" + initX);
            }
        }
    }
}
