package com.pt.ml.util

import com.pt.ml.visualization.Histogram
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

object MultiClassEvaluation {
    def accuracy(spark: SparkSession, preAndLabel: RDD[(Double, Double)]): Double = {
        val dataSet = spark.createDataFrame(preAndLabel).toDF("pre", "label")
        accuracy(dataSet, "pre", "label")
    }

    def weightedPrecision(spark: SparkSession, preAndLabel: RDD[(Double, Double)]): Double = {
        val dataSet = spark.createDataFrame(preAndLabel).toDF("pre", "label")
        weightedPrecision(dataSet, "pre", "label")
    }

    def weightedRecall(spark: SparkSession, preAndLabel: RDD[(Double, Double)]): Double = {
        val dataSet = spark.createDataFrame(preAndLabel).toDF("pre", "label")
        weightedRecall(dataSet, "pre", "label")
    }

    def f1(spark: SparkSession, preAndLabel: RDD[(Double, Double)]): Double = {
        val dataSet = spark.createDataFrame(preAndLabel).toDF("pre", "label")
        f1(dataSet, "pre", "label")
    }

    def accuracy(dataSet: Dataset[_], pre: String, label: String): Double = {
        getMetric(dataSet, pre, label, "accuracy")
    }

    def weightedPrecision(dataSet: Dataset[_], pre: String, label: String): Double = {
        getMetric(dataSet, pre, label, "weightedPrecision")
    }

    def weightedRecall(dataSet: Dataset[_], pre: String, label: String): Double = {
        getMetric(dataSet, pre, label, "weightedRecall")
    }

    def f1(dataSet: Dataset[_], pre: String, label: String): Double = {
        getMetric(dataSet, pre, label, "f1")
    }

    private def getMetric(dataSet: Dataset[_], pre: String, label: String, metricName: String): Double = {
        val evaluator = new MulticlassClassificationEvaluator()
                .setPredictionCol(pre)
                .setLabelCol(label)
                .setMetricName(metricName)
        evaluator.evaluate(dataSet)
    }

    //查看每一类的precision 和 recall
    def showPrecisionRecallF1Tpr(preLabel: RDD[(Double, Double)]): Unit = {
        val metrics = new MulticlassMetrics(preLabel)
        val labels = metrics.labels
        val pValues = new Array[Double](labels.length)
        val rValues = new Array[Double](labels.length)
        val f1Values = new Array[Double](labels.length)
        labels.zipWithIndex.foreach { l =>
            pValues(l._2) = metrics.precision(l._1)
            rValues(l._2) = metrics.recall(l._1)
            f1Values(l._2) = metrics.fMeasure(l._1)
        }
        val histogram = new Histogram("多分类指标可视化", "precision-recall-f1")
        histogram.setDatasets("precision", labels.map(v => v.toString), pValues)
        histogram.setDatasets("recall", labels.map(v => v.toString), rValues)
        histogram.setDatasets("f1", labels.map(v => v.toString), f1Values)
        histogram.showPlot()
    }

    def showPrecisionRecallF1Tpr(dataSet: Dataset[_], pre: String, label: String): Unit = {
        val preLabel = dataSet.select(pre, label).toDF("pre", "label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
        showPrecisionRecallF1Tpr(preLabel)
    }
}
