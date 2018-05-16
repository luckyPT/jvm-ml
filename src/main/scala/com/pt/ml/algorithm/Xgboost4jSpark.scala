package com.pt.ml.algorithm

import com.pt.ml.util.MultiClassEvaluation
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * xgboost对于spark版本有要求，使用dataFrame存在兼容性问题，因此使用Rdd
  * 暂时还没调好，十几分钟内执行不完
  */
object Xgboost4jSpark {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[8]")
                //.config("spark.task.cpus", "8") 切记不可增加此配置
                .getOrCreate()
        spark.sparkContext.setLogLevel("INFO")

        val ministData = spark.read.format("csv").option("header", "true")
                .load("/home/panteng/文档/dataset/MINIST/train.csv").rdd
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
                        LabeledPoint(row.getAs[String](0).toDouble, org.apache.spark.ml.linalg.Vectors.sparse(row.length - 1, indeices.toArray, values.toArray))
                }

        val Array(trainData, testData) = ministData.randomSplit(Array(0.7, 0.3))
        val numRound = 10
        val paramMap = List(
            "booster" -> "gbtree",
            "objective" -> "multi:softmax",
            "num_class" -> 10,
            "eta" -> 0.1f,
            "gamma" -> 0.05,
            "max_depth" -> 8,
            "silent" -> 1
        ).toMap

        val model = XGBoost.trainWithRDD(trainData, paramMap, numRound, trainData.partitions.length)
        print("train success")

        val pre = model.predict(testData.map(l => l.features)).collect()
        val testMinistData = testData.collect()
        val preStatistics = mutable.ListBuffer[(Float, Float)]()
        for (i <- pre.indices) {
            val label = testMinistData(i).label
            preStatistics.append((pre(i)(0), label.toFloat))
        }
        val preAndLabel = spark.sparkContext.parallelize(preStatistics).map {
            case (f1, f2) =>
                (f1.toDouble, f2.toDouble)
        }
        println(s" test accuracy:" + MultiClassEvaluation.accuracy(spark, preAndLabel))
    }
}
