package com.pt.ml.algorithm

import com.pt.ml.util.BinaryClassEvaluation
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

/**
  * spark 只提供了线性 SVM，即只有线性核函数；不支持其他核函数
  * SVM调优：
  * 调整惩罚系数C（分错之后的惩罚，与L2惩罚并不是同一个概念）
  * 变换核函数,调整核函数的相关参数和
  */
object SurportVectorMerchine {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[8]")
                //.config("spark.task.cpus", "8") 切记不可增加此配置
                .getOrCreate()
        spark.sparkContext.setLogLevel("INFO")
        import spark.implicits._
        val ministData = spark.read.format("csv").option("header", "true")
                .load("/home/panteng/文档/dataset/MINIST/train.csv")
                .map {
                    row =>
                        val indeices = new ListBuffer[Int]()
                        val values = new ListBuffer[Double]()
                        var value = 0
                        for (i <- Range(1, row.length)) {
                            value = row.getAs[String](i).toInt
                            if (value > 40) {
                                indeices += i
                                values += 1
                            }
                        }
                        (row.getAs[String](0).toDouble, org.apache.spark.ml.linalg.Vectors.sparse(row.length - 1, indeices.toArray, values.toArray))
                }
                .toDF("label", "features")
                .filter($"label" < 2)

        val Array(trainData, testData) = ministData.randomSplit(Array(0.7, 0.3))
        testData.show(1000)
        val lsvc = new LinearSVC()
                .setMaxIter(10)
                .setRegParam(0.1)

        val lsvcModel = lsvc.fit(trainData)
        println(s"Coefficients: ${lsvcModel.coefficients.size}.s Intercept: ${lsvcModel.intercept}")
        val pre = lsvcModel.transform(testData)
        pre.show(1000)
        val preRdd = pre.rdd.map(row => (1.0 / (1 + math.pow(Math.E, row.getAs[DenseVector](2).apply(0))), row.getDouble(0)))
        println(preRdd.take(1000).mkString("\n"))
        BinaryClassEvaluation.showThresholdPrecisionRecallCurve(preRdd)
    }
}
