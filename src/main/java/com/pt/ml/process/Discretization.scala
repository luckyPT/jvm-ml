package com.pt.ml.process

import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.SparkSession

/**
  * 变量离散化（跟oneHot有相似之处）
  */
object Discretization {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()

        val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)
        val data = Array(-999.9, -0.5, -0.3, 0.0, 0.2, 999.9, 500.2)

        val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
        val bucketizer = new Bucketizer()
                .setInputCol("features")
                .setOutputCol("bucketedFeatures")
                .setSplits(splits)

        val bucketedData = bucketizer.transform(dataFrame)

        println(s"Bucketizer output with ${bucketizer.getSplits.length - 1} buckets")
        bucketedData.show()
    }
}
