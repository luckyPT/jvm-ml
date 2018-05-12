package com.pt.ml.algorithm

import com.pt.ml.util.MultiClassEvaluation
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession

/**
  * 跟单棵树相比一个主要目的是为了防止过拟合,增强泛化能力;
  * 可以用于分类和回归；既适用于离散特征也适用于连续特征;
  * 每一棵树的训练都是独立的，所以可以并发进行；目标函数是将各个树的预测进行组合以减少预测集上的方差
  * 随机性包含：
  * - 每次迭代，都会在数据集上随机采样数据
  * - 每一棵树节点的分裂，在一个随机特征集上进行
  * 预测过程，回归和分类有所不同：
  * - 分类：所有树进行投票，每个树一票；得票多的类为最终预测类
  * - 回归：各个树预测值，求平均
  * 调优：
  * - 森林中树的数目
  * - 每棵树的深度
  * - subsamplingRate(实际测试中发现1是最好的)
  * - featureSubsetStrategy
  */
object RandomForest {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()
        val iris = spark.read.format("libsvm").load("/home/panteng/文档/dataset/iris.libsvm")
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

        val rf = new RandomForestClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures")
                //.setImpurity("gini")
                .setNumTrees(1)

        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels)

        // Chain indexers and forest in a Pipeline.
        val pipeline = new Pipeline()
                .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainData)

        // Make predictions.
        val predictions = model.transform(testData)
        predictions.show(false)
        val accuracy = MultiClassEvaluation.accuracy(predictions, "indexedLabel", "prediction")
        val weightPrecision = MultiClassEvaluation.weightedPrecision(predictions, "indexedLabel", "prediction")
        val weightRecall = MultiClassEvaluation.weightedRecall(predictions, "indexedLabel", "prediction")
        val f1 = MultiClassEvaluation.f1(predictions, "indexedLabel", "prediction")
        println(s"test result:accuracy = $accuracy; weightPrecision = $weightPrecision; weightRecall = $weightRecall; f1 = $f1")
        MultiClassEvaluation.showPrecisionRecallF1Tpr(predictions, "indexedLabel", "prediction")
        val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
        println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
    }
}
