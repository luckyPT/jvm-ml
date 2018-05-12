package com.pt.ml.process

import java.awt.Color

import com.pt.ml.visualization.Scatter
import org.apache.spark.ml.feature.PCA
import org.apache.spark.sql.SparkSession

/**
  * PCA降维(原始最大维度不能超过65535)
  * 使用正交变换，将一组看起来相关性强的特征，转为一组没有线性关系的特征（成为主成成分）
  * 协方差矩阵：协方差(i,j)=（第i列的所有元素-第i列的均值）*（第j列的所有元素-第j列的均值）
  * 特征值&特征向量：X是一个矩阵，v是一个非零向量，m是一个常数，满足 Xv = mv 则v、m的X的特征向量和特征值
  * 计算过程如下：
  * - 计算矩阵的协方差矩阵，并计算协方差矩阵的特征值和特征向量
  * - 取特征值最大的k特征向量，将原矩阵与特征向量矩阵相乘得到变换后的矩阵
  * 另一种计算方式：
  * - 计算矩阵的协方差矩阵
  * - 对协方差矩阵进行奇异值分解，得到一个矩阵和一个向量
  * - 将得到的矩阵和和向量进行一定的变换,作为PCA model的参数
  * 用途：
  * PCA一般用来降维 可视化；很多数据压缩算法也用到了PCA降维；
  * 在有监督学习中如果特征太多，也可以进行小幅度的降维，但不建议大幅度降维；
  */
object Pca {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()
        //构建训练数据
        val iris = spark.read.format("libsvm").load("/home/panteng/文档/dataset/iris.libsvm")
        val pca = new PCA()
                .setInputCol("features")
                .setOutputCol("pcaFeatures")
                .setK(2)
                .fit(iris)

        val result = pca.transform(iris).select("label", "pcaFeatures").rdd
                .map {
                    row =>
                        (row.getDouble(0), row.getAs[org.apache.spark.ml.linalg.Vector](1).toArray)
                }
                .collect().groupBy[Double](_._1)

        val category = result.map {
            case (label, xyPoints) =>
                val x = new Array[Double](xyPoints.length)
                val y = new Array[Double](xyPoints.length)
                val points = xyPoints.map(i => i._2)
                points.zipWithIndex.foreach {
                    case (point, index) =>
                        x(index) = point(0)
                        y(index) = point(1)
                }
                (label, x, y)

        }
        val scatter = new Scatter("PCA降维", "鸢尾花数据降维可视化")
        category.foreach {
            case (label, x, y) =>
                scatter.addData(label + "", x, y, if (label == 0) Color.BLACK else if (label == 1) Color.RED else Color.GREEN)
        }
        scatter.showPlot()
    }
}
