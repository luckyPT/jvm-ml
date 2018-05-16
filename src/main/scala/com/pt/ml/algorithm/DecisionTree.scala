package com.pt.ml.algorithm

import com.pt.ml.util.MultiClassEvaluation
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession

/**
  * 分类树（spark貌似没有提供后剪枝的实现）
  * 使用决策树可以不对数据进行标准化和归一化
  * 尤其使适用于非线性关系，特征之间相互作用的情况
  * 每一次的分裂都是为了降低总体的纯度（impurity）
  * 调优：
  * impurity：误差衡量标准（熵、GINI系数、均方差）对分类来说，熵比基尼系数好一些
  * 提前停止分裂，防止过拟合：
  * maxDepth - 最大深度(SPARK 限制必须不大于30)；
  * minInstancesPerNode - 节点样本最小数目
  * minInfoGain - 分裂之后，误差降幅最小值
  * maxBins:对于连续特征，最大分段数量
  * maxMemoryInMB：训练使用内存
  * subsamplingRate：训练数据的采样比例，对于集成决策树很有帮助，对于单棵树没有太大帮助
  */
object DecisionTree {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()
        //构建训练数据
        val iris = spark.read.format("libsvm").load("/home/panteng/文档/dataset/iris.libsvm")
        val dataSplit = iris.randomSplit(Array(0.7, 0.3))
        val trainData = dataSplit(0)
        val testData = dataSplit(1)

        //将String 类型的label转为index表示
        val labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(trainData)

        //某些情况下，可以自动区分连续特征和离散特征，并自动对离散特征编码
        val featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4) // 拥有大于4个不同的值即认为是连续型特征
                .fit(trainData)

        val dt = new DecisionTreeClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures")
                .setImpurity("gini") //entropy 或 gini

        //标签反转，由index转为最初的String
        val labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels)

        val pipeline = new Pipeline()
                .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
        val model = pipeline.fit(trainData)
        val predictions = model.transform(testData)
        predictions.show(false)

        val accuracy = MultiClassEvaluation.accuracy(predictions, "indexedLabel", "prediction")
        val weightPrecision = MultiClassEvaluation.weightedPrecision(predictions, "indexedLabel", "prediction")
        val weightRecall = MultiClassEvaluation.weightedRecall(predictions, "indexedLabel", "prediction")
        val f1 = MultiClassEvaluation.f1(predictions, "indexedLabel", "prediction")
        println(s"test result:accuracy = $accuracy; weightPrecision = $weightPrecision; weightRecall = $weightRecall; f1 = $f1")
        MultiClassEvaluation.showPrecisionRecallF1(predictions, "indexedLabel", "prediction")
        val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
        println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    }
}
