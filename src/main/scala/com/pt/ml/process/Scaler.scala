package com.pt.ml.process

import org.apache.spark.ml.feature.{MinMaxScaler, Normalizer, StandardScaler}
import org.apache.spark.sql.SparkSession

/**
  * 关于标准化（standardized）、归一化（Normalized）、缩放（scaler）
  * 个人理解：
  * 标准化 就是指 减均值除以方差的变换，可以把标准化当作scaler的一种
  * scaler 就是特征缩放，有好多种缩放方式，如：minMaxScaler、maxAbsScaler
  * 归一化 从spark的实现来看，就是将向量变换城单位向量，解决不同特征值差别巨大，导致收敛慢，精度低的问题
  */
object Scaler {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()
        val iris = spark.read.format("libsvm").load("/home/panteng/文档/dataset/iris.libsvm")
        iris.show(10, truncate = false)

        //标准化(减均值 除以 标准差)
        val standardScaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("standerdFeatures")
                .setWithStd(true)
                .setWithMean(true)
        val scalerModel = standardScaler.fit(iris)
        val standardIris = scalerModel.transform(dataset = iris)
        standardIris.show(10, truncate = false)
        //mixMax缩放 计算方式：((e_i - E_min)/(E_max - E_min))/(max - min) + min
        //其中E_*为对应的统计值，e当前待转换的值，max设定的最大值 min设定的最小值
        val minMaxScaler = new MinMaxScaler()
                .setMin(-1)
                .setMax(1)
                .setInputCol("features")
                .setOutputCol("minMaxFeatures")
        val minMaxScalerModel = minMaxScaler.fit(iris)
        val minMaxScalerIris = minMaxScalerModel.transform(iris)
        minMaxScalerIris.show(150, truncate = false)

        //归一化 使得每个vector都是单位向量
        val normalizer = new Normalizer()
                .setInputCol("standerdFeatures")
                .setOutputCol("normFeatures")
                .setP(1.0)
        val l1NormData = normalizer.transform(standardIris)
        l1NormData.show(false)
    }
}
