package com.pt.ml.util

import java.awt.Color

import com.pt.ml.visualization.Line
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

object BinaryClassEvaluation {
    def showRocCurve(preLabel: RDD[(Double, Double)]): Unit = {
        val metrics = new BinaryClassificationMetrics(preLabel)
        val rocPoint = metrics.roc().collect()
        val x = new Array[Double](rocPoint.length)
        val y = new Array[Double](rocPoint.length)
        rocPoint.zipWithIndex.foreach {
            case (point, index) =>
                x(index) = point._1
                y(index) = point._2
        }

        val auc = metrics.areaUnderROC()
        val rocCurve = new Line("ROC曲线", s"AUC=$auc")
        rocCurve.linePlot("roc", x, y)
        rocCurve.showPlot()
    }

    def showPrecisionRecallCurve(preLabel: RDD[(Double, Double)]): Unit = {
        val metrics = new BinaryClassificationMetrics(preLabel)
        val precision = metrics.precisionByThreshold.collect()
        val recall = metrics.recallByThreshold.collect()
        val p_x = new Array[Double](precision.length)
        val p_y = new Array[Double](precision.length)
        precision.zipWithIndex.foreach {
            case (point, index) =>
                p_x(index) = point._1
                p_y(index) = point._2
        }

        val r_x = new Array[Double](recall.length)
        val r_y = new Array[Double](recall.length)
        recall.zipWithIndex.foreach {
            case (point, index) =>
                r_x(index) = point._1
                r_y(index) = point._2
        }

        val prCurve = new Line("PR-曲线", "PR-曲线")
        prCurve.linePlot("precision", p_x, p_y)
        prCurve.linePlot("recall", r_x, r_y, Color.BLACK, 2)
        prCurve.showPlot()
    }

    def showF1Curve(preLabel: RDD[(Double, Double)]): Unit = {
        val metrics = new BinaryClassificationMetrics(preLabel)
        val f1Score = metrics.fMeasureByThreshold.collect()

        val x = new Array[Double](f1Score.length)
        val y = new Array[Double](f1Score.length)
        f1Score.zipWithIndex.foreach {
            case (point, index) =>
                x(index) = point._1
                y(index) = point._2
        }

        val F1Curve = new Line("F1曲线", "F1曲线")
        F1Curve.linePlot("F1", x, y)
        F1Curve.showPlot()
    }

}
