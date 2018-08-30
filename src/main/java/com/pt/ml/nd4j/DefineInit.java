package com.pt.ml.nd4j;

import org.apache.spark.sql.sources.In;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class DefineInit {
    public static void main(String[] args) {
        //初始化0矩阵
        INDArray zeroArray = Nd4j.zeros(3, 5);
        System.out.println("zeroArray:" + zeroArray);
        //初始化1矩阵
        INDArray oneArray = Nd4j.ones(3, 5);
        System.out.println("oneArray:" + oneArray);
        //初始化随机矩阵
        INDArray randomArray = Nd4j.rand(3, 5);
        System.out.println("randomArray:" + randomArray);
        //根据数组创建矩阵
        double[][] array = new double[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        INDArray array2Vec = Nd4j.create(array);
        System.out.println("array2Vec:" + array2Vec);

        //获取基本信息 行数、列数、shape、维度、元素个数
        System.out.println("row count:" + oneArray.rows());
        System.out.println("column count:" + oneArray.columns());
        System.out.println("shape:" + Arrays.toString(oneArray.shape()));
        System.out.println("dimensions:" + oneArray.rank());
        System.out.println("element count:" + oneArray.length());

        //INDArray 的分类；scalar vector matrix
        INDArray scalar = Nd4j.scalar(0.2);
        INDArray vector = Nd4j.trueVector(new float[] {0.1F, 0.2F, 0.3F});
        System.out.println("scalar=" + scalar.isScalar());
        System.out.println("vector=" + vector.isVector());
        System.out.println("matrix=" + oneArray.isMatrix());

        //高维矩阵
        INDArray highDimensionsMatrix = Nd4j.create(5, 4, 3, 2);//元组值为0
        INDArray highDimensionsMatrix1 = Nd4j.ones(5, 4, 3, 2);//元素值为1
        INDArray highDimensionsMatrix2 = Nd4j.rand(new int[] {5, 4, 2, 2, 1});//最后一维是1或者略去，是有区别的
        System.out.println("highDimensionsMatrix:" + Arrays.toString(highDimensionsMatrix2.shape()));
        System.out.println("highDimensionsMatrix:" + highDimensionsMatrix2);

        //稀疏矩阵
        //二维
        float[] data = new float[] {0.1F, 0.2F};
        int[][] indices = new int[][] {{1, 1}, {2, 2}};
        int[] shape = new int[] {3, 5};
        INDArray sparseArray = Nd4j.createSparseCOO(data, indices, shape);
        System.out.println("isSparse:" + sparseArray.isSparse());
        System.out.println(sparseArray.toDense());
        
    }
}
