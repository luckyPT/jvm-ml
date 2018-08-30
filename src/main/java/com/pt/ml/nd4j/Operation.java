package com.pt.ml.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Operation {
    public static void main(String[] args) {
        INDArray highDimensionsMatrix = Nd4j.rand(new int[] {3, 3, 3, 2});
        System.out.println("原始：" + highDimensionsMatrix);//这里输出值是精确到某一位，并不是准确值
        //读取某个元素
        float el = highDimensionsMatrix.getFloat(new int[] {1, 0, 2, 0});
        System.out.println("el:" + el);
        //读取某个维度,可以将任意维度的INDArray 理解为二维的，只不过元素不再是一个数字，而是INDArray
        INDArray row = highDimensionsMatrix.getRow(0);
        INDArray column = highDimensionsMatrix.getColumn(0);
        System.out.println("oneDimension:" + column);
        //修改某个元素
        highDimensionsMatrix.putScalar(new int[] {0, 0, 0, 0}, 1.99);
        System.out.println("modify one el:" + highDimensionsMatrix);
        //修改某一个维度
        highDimensionsMatrix.getRow(0).getRow(0).putRow(0, Nd4j.ones(1, 2));
        System.out.println("modify one IDArray:" + highDimensionsMatrix);

        //广播 加、减、乘、除 add和addi的区别：前者对象本身不变，返回加1之后的对象；后者在自身加1
        highDimensionsMatrix.addi(1);
        highDimensionsMatrix.muli(10);
        System.out.println("add 1:" + highDimensionsMatrix);

        //矩阵的加、减、乘、除
        INDArray one = Nd4j.ones(3, 3, 2).muli(0.5);
        one.getColumn(0).putRow(0, Nd4j.create(new double[] {0.5, 1.5}));

        INDArray two = Nd4j.ones(3, 3, 2).muli(2);
        two.getColumn(0).putRow(0, Nd4j.create(new double[] {2, 3}));
        System.out.println("one:" + one);
        System.out.println("two:" + two);

        INDArray add = one.add(two);
        System.out.println("add:" + add);
        INDArray multi = one.muli(two);
        System.out.println("multi:" + multi);

        //Vec的加减乘除
        INDArray vec1 = Nd4j.trueVector(new double[] {1, 2, 3});
        INDArray vec2 = Nd4j.trueVector(new double[] {4, 5, 6});
        INDArray vecAdd = vec1.add(vec2);
        INDArray vecMulti = vec1.mul(vec2);
        System.out.println("vecAdd:" + vecAdd); //对应元素相加
        System.out.println("vecMulti:" + vecMulti);//对应元素相乘

        //vec 和 矩阵 运算异常

        //测试性能(1w维向量相乘，循环10000次，耗时5秒左右)
        INDArray matrix1 = Nd4j.rand(200, 1).muli(2);
        INDArray matrix2 = Nd4j.rand(1, 200).muli(10);
        long start = System.currentTimeMillis();
        for (int i = 0; i < 10000; i++) {
            matrix1.mul(matrix2);
        }
        System.out.println("1000 × 1000 耗时：" + (System.currentTimeMillis() - start) + " ms");
    }
}
