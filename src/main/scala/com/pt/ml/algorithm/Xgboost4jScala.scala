package com.pt.ml.algorithm

import java.io.{File, PrintWriter}

import com.pt.ml.util.MultiClassEvaluation
import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}
import org.apache.spark.sql.SparkSession

import scala.io.Source
import scala.collection.JavaConversions._
import scala.collection.mutable

/**
  * xgboost是一种集成学习算法;
  * 经过对树的深度和迭代次数进行调参，在MINIST数据集上可以做到accracy=0.97
  * 优点:
  * - 可以自定义损失函数
  * - 在损失函数上增加了正则化，可以有效防止过拟合；
  * - 并且对损失函数进行了变形，使用二阶泰勒展开进行近似，收敛速度更快;
  * - 缺失值的处理，实际上是假设将缺失值样本放进左子树和右子树，分别计算gain
  */
object Xgboost4jScala {
    def main(args: Array[String]): Unit = {
        val ministData = Source.fromFile("./dataset/MINIST/train.csv").getLines()
                .filter(str => !str.contains("pixel0"))
                .map {
                    line =>
                        val array = line.split(",").toList
                        val label = array.head.toFloat
                        val values = array.subList(1, array.length).map(str => str.toFloat).toArray

                        new LabeledPoint(label, null, values)
                }
        val submissionMinistData = Source.fromFile("./dataset/MINIST/test.csv").getLines()
                .filter(str => !str.contains("pixel0"))
                .map {
                    line =>
                        val array = line.split(",").toList
                        val label = array.head.toFloat
                        val values = array.subList(1, array.length).map(str => str.toFloat).toArray
                        new LabeledPoint(label, null, values)
                }.toList

        val testCount = 5000
        val testMinistData = ministData.slice(0, testCount).toList
        val trainMinistData = ministData.drop(testCount).toList

        val trainData = new DMatrix(trainMinistData.iterator)
        val testData = new DMatrix(testMinistData.iterator)

        // number of iterations
        val paramMap = List(
            "booster" -> "gbtree",
            "objective" -> "multi:softmax",
            "num_class" -> 10,
            "eta" -> 0.1,
            "gamma" -> 0.05,
            "max_depth" -> 8 //在迭代次数为100的时候，最优值为8
        ).toMap

        val round = 100 //迭代次数为100的时候，接近最佳值
        // train the model
        println("start train... ...")
        val model = XGBoost.train(trainData, paramMap, round)

        // 使用spark进行统计
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        // run prediction
        val trainError = model.predict(trainData)
        val trainStatistics = mutable.ListBuffer[(Float, Float)]()
        for (i <- trainError.indices) {
            val label = trainMinistData.get(i).label
            trainStatistics.append((trainError(i)(0), label))
        }
        val trainPreAndLabel = spark.sparkContext.parallelize(trainStatistics).map {
            case (f1, f2) =>
                (f1.toDouble, f2.toDouble)
        }
        println(s" train accuracy:" + MultiClassEvaluation.accuracy(spark, trainPreAndLabel))
        //test
        val pre = model.predict(testData)
        val preStatistics = mutable.ListBuffer[(Float, Float)]()
        for (i <- pre.indices) {
            val label = testMinistData.get(i).label
            preStatistics.append((pre(i)(0), label))
        }
        val preAndLabel = spark.sparkContext.parallelize(preStatistics).map {
            case (f1, f2) =>
                (f1.toDouble, f2.toDouble)
        }
        println(s" test accuracy:" + MultiClassEvaluation.accuracy(spark, preAndLabel))
        //submission
        val subPre = model.predict(new DMatrix(submissionMinistData.iterator))
        val writer = new PrintWriter(new File("submission.txt"))
        for (i <- subPre.indices) {
            writer.write((i + 1) + "," +
                    (if (Math.round(subPre(i)(0)) < 0) 0 else if (Math.round(subPre(i)(0)) > 9) 9 else Math.round(subPre(i)(0))) +
                    "\n")
        }
        writer.flush()
        writer.close()
        println("---complete----")
    }
}
