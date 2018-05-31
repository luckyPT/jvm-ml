package com.pt.ml.algorithm

import com.pt.ml.util.{BinaryClassEvaluation, MultiClassEvaluation}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, GBTClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession

/**
  * spark提供的GBDT 只能解决二分类和回归问题
  * 目的是更小的降低loss
  */
object GradientBoostTree {
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
        val dataSplit = iris.randomSplit(Array(0.7, 0.3))
        val trainData = dataSplit(0)
        val testData = dataSplit(1)
        val labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(trainData)

        val featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(trainData)

        // Train a GBT model.
        val gbt = new GBTClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures")
                .setMaxIter(10)
                .setFeatureSubsetStrategy("auto")

        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels)

        // Chain indexers and GBT in a Pipeline.
        val pipeline = new Pipeline()
                .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainData)

        // Make predictions.
        val predictions = model.transform(testData)
        predictions.show(false)
        val preAndLabel = predictions.select($"probability", $"label")
                .toDF().rdd
                .map {
                    row =>
                        val pre = row.getAs[org.apache.spark.ml.linalg.Vector](0).toArray(1)
                        (pre, row.getDouble(1))

                }.cache()
        BinaryClassEvaluation.showRocCurve(preAndLabel)
        BinaryClassEvaluation.showThresholdPrecisionRecallCurve(preAndLabel)
        BinaryClassEvaluation.showF1Curve(preAndLabel)

    }
}
