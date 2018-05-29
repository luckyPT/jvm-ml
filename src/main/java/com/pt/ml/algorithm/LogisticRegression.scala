package com.pt.ml.algorithm

import com.pt.ml.util.BinaryClassEvaluation
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

/**
  * 标签最好是从0开始，依次递增
  * 调优：
  * 数据标准化，归一化
  * 正则化参数
  * 优化方法
  */
object LogisticRegression {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()
        import spark.implicits._
        //构建训练数据
        val iris = spark.read.format("libsvm").load("/home/panteng/文档/dataset/iris.libsvm")
                .filter(r => r.getDouble(0) > 0.5) //只取label = 1和2的两种花
                .select($"label" - 1.0, $"features") //标签从0开始
                .toDF("label", "features")
                .randomSplit(Array(0.7, 0.3)) //随机分割为两部分，作为训练集和测试集

        val trainData = iris(0)
        val testData = iris(1)
        println(s"train count:${trainData.count()} testCount:${testData.count()}")
        trainData.show(false)

        //构建模型，训练
        val lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
        //.setFamily("multinomial") //多项回归，输出的是权重是一个矩阵；逻辑回归，输出的是一个向量；
        val lrModel = lr.fit(trainData)
        println(s"Coefficients: ${lrModel.coefficientMatrix} Intercept: ${lrModel.interceptVector}")
        val trainSummary = lrModel.binarySummary
        val objectiveHistory = trainSummary.objectiveHistory
        println("objectiveHistory")
        objectiveHistory.foreach(println)
        val roc = trainSummary.roc
        println("train ROC:")
        roc.show()
        println(s"train AUC: ${trainSummary.areaUnderROC}")

        //预测
        val pre = lrModel.transform(trainData).cache()
        pre.show(false)
        val preAndLabel = pre.select($"probability", $"label")
                .toDF().rdd
                .map {
                    row =>
                        val pre = row.getAs[org.apache.spark.ml.linalg.Vector](0).toArray(1)
                        (pre, row.getDouble(1))

                }.cache()
        println(preAndLabel.take(20).mkString("\n"))

        BinaryClassEvaluation.showRocCurve(preAndLabel)
        BinaryClassEvaluation.showPrecisionRecallCurve(preAndLabel)
        BinaryClassEvaluation.showF1Curve(preAndLabel)
    }

}
