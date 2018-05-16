package com.pt.ml.example

import com.pt.ml.util.MultiClassEvaluation
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

object MinistClassifier {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        import spark.implicits._
        //构建训练数据
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
                }.toDF("label", "features")

        import spark.implicits._
        val testData = spark.read.format("csv").option("header", "true")
                .load("/home/panteng/文档/dataset/MINIST/test.csv")
                .map {
                    row =>
                        val indeices = new ListBuffer[Int]()
                        val values = new ListBuffer[Double]()
                        var value = 0
                        for (i <- Range(0, row.length)) {
                            value = row.getAs[String](i).toInt
                            if (value > 40) {
                                indeices += i
                                values += 1
                            }
                        }
                        //很奇葩，必须返回元组才能toDF
                        (1, org.apache.spark.ml.linalg.Vectors.sparse(row.length, indeices.toArray, values.toArray))
                }.toDF("label", "features").select("features")

        val standardScaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("standardFeatures")
                .setWithStd(true)
                .setWithMean(true)

        val Array(trainMinist, testMinist) = ministData.randomSplit(Array(0.7, 0.3))
        //标准化 对于逻辑回归稍微有点效果
        /*val scalerModel = standardScaler.fit(trainMinist)
        val trainMinistStandard = scalerModel.transform(trainMinist)
        val testMinistStandard = scalerModel.transform(testMinist)*/
        val model = gbdtClassfier(spark, trainMinist, testMinist)

        /*val test = model.transform(testData)
        test.select("prediction").rdd.zipWithIndex()
                .map(r => Seq(r._2 + 1, r._1.getDouble(0).toInt).mkString(","))
                .repartition(1).saveAsTextFile("/home/panteng/文档/dataset/MINIST/submission")*/

    }

    /**
      * accuracy最优值为0.92
      *
      * @param trainMinistStandard 训练集
      * @param testMinistStandard  测试集
      */
    def logistMultiClassfier(trainMinistStandard: DataFrame, testMinistStandard: DataFrame): LogisticRegressionModel = {
        val lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.001)
                .setElasticNetParam(0.1)
                .setFeaturesCol("standardFeatures")
                .setFamily("multinomial") //多项回归，输出的是权重是一个矩阵；逻辑回归，输出的是一个向量；

        val lrModel = lr.fit(trainMinistStandard)
        lrModel.setFeaturesCol("standardFeatures")

        val pre = lrModel.transform(testMinistStandard)
        println("accuracy:" + MultiClassEvaluation.accuracy(pre, "prediction", "label"))
        lrModel
    }

    /**
      * 使用单棵决策树，accuracy最优值为0.86（过拟合）；通过调节最大深度、叶节点最小样本数、最小增益等无法解决过拟合问题
      * 使用随机森林解决过拟合问题：accuracy最优值0.95（19棵树）
      * 可以使用单棵树确定树的深度和最小样本数
      *
      * @param trainMinistStandard 训练集
      * @param testMinistStandard  测试集
      */
    def randomForestClassfier(trainMinistStandard: DataFrame, testMinistStandard: DataFrame): RandomForestClassificationModel = {
        val rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("entropy")
                .setMinInstancesPerNode(3)
                .setMaxDepth(15)
                .setFeatureSubsetStrategy("0.1")
                .setNumTrees(19)

        val randomForestModel = rf.fit(trainMinistStandard)
        val preTrain = randomForestModel.transform(trainMinistStandard)
        val preTest = randomForestModel.transform(testMinistStandard)
        println(s" preTrain accuracy:" + MultiClassEvaluation.accuracy(preTrain, "prediction", "label"))
        println(s" preTest accuracy:" + MultiClassEvaluation.accuracy(preTest, "prediction", "label"))
        randomForestModel
    }

    /**
      * spark-gbdt只能做二分类，这里使用gbdt回归来进行预测，之后进行取整
      * 效果不理想0.82左右
      *
      * @param spark               spark上下文
      * @param trainMinistStandard 训练集
      * @param testMinistStandard  测试集
      * @return
      */
    def gbdtRegression(spark: SparkSession, trainMinistStandard: DataFrame, testMinistStandard: DataFrame): GBTRegressionModel = {

        val gbt = new GBTRegressor()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setMaxDepth(26)
                .setMaxIter(10)
                .setFeatureSubsetStrategy("0.2")

        val gbdtModel = gbt.fit(trainMinistStandard)
        val preTrain = gbdtModel.transform(trainMinistStandard).select("label", "prediction")
                .rdd
                .map {
                    row =>
                        (row.getDouble(0), Math.rint(row.getDouble(1)))
                }

        val preTest = gbdtModel.transform(testMinistStandard).select("label", "prediction")
                .rdd
                .map {
                    row =>
                        (row.getDouble(0), Math.rint(row.getDouble(1)))
                }
        println(s" preTrain accuracy:" + MultiClassEvaluation.accuracy(spark, preTrain))
        println(s" preTest accuracy:" + MultiClassEvaluation.accuracy(spark, preTest))
        gbdtModel

    }

    /**
      * 使用gbdt二分类器结合OneVsRest进行多分类
      * OneVsRest 存在的问题：二分类时的数据不均衡；每一类别的置信度大小比较并不十分严谨；训练非常耗时;
      * 效果还可以，accuracy：0.94
      *
      * @param spark               spark上下文
      * @param trainMinistStandard 训练集
      * @param testMinistStandard  测试集
      * @return
      */
    def gbdtClassfier(spark: SparkSession, trainMinistStandard: DataFrame, testMinistStandard: DataFrame): OneVsRestModel = {
        // Train a GBT model.
        val gbt = new GBTClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures")
                .setMaxDepth(26)
                .setMaxIter(10)
                .setFeatureSubsetStrategy("0.2")

        val ovr = new OneVsRest().setClassifier(gbt)
        val ovrModel = ovr.fit(trainMinistStandard)

        val preTrain = ovrModel.transform(trainMinistStandard).select("label", "prediction")
                .rdd
                .map {
                    row =>
                        (row.getDouble(0), Math.rint(row.getDouble(1)))
                }

        val preTest = ovrModel.transform(testMinistStandard).select("label", "prediction")
                .rdd
                .map {
                    row =>
                        (row.getDouble(0), Math.rint(row.getDouble(1)))
                }
        println(s" preTrain accuracy:" + MultiClassEvaluation.accuracy(spark, preTrain))
        println(s" preTest accuracy:" + MultiClassEvaluation.accuracy(spark, preTest))
        ovrModel
    }
}
