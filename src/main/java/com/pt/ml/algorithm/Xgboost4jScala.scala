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
  * 参数 http://xgboost.readthedocs.io/en/latest/parameter.html：
  * - general：
  * -- booster：gbtree、gblinear、dart
  * -- silent：0、1 是否输出日志
  * -- nthread：线程数目，默认是最大可用值
  * - Boost：
  * -- eta(learning_rate)：默认0.3,降低新树叶子输出值权重，防止过拟合
  * -- gamma：[0,∞],节点分裂的惩罚系数，值越大，模型则越保守
  * -- max_depth：
  * -- min_child_weight：计算节点中每个样本的某个值（暂时不清楚是什么值），然后求和；如果这个值小于min_child_weight则这个节点不再继续分裂；
  * -- max_delta_step：限制每棵树权重改变的最大步长，通常不需要设定，在逻辑回归中，如果正负例不均衡，将这个值设为1-10之间或许有帮助
  * -- subsample： (0,1]训练降采样比例
  * -- colsample_bytree：(0,1]构建每棵树使用特征的比例
  * -- lambda：L2 正则化参数
  * -- alpha：L1 正则化参数
  * -- tree_method
  * -- scale_pos_weight：设置为 sum(Negative)/sum(Positive)均衡正负样本比例
  * -- 当tree_method设置为hist时可以设置以下参数（其他值不可以）：
  * --- grow_policy
  * --- max_leaves：当grow_policy设置为lossguide时生效
  * --- max_bin
  * - Learning Task Parameters：
  * -- objective：reg:linear、reg:logistic、binary:logistic、binary:logitraw、count:poisson、survival:cox、multi:softmax、multi:softprob等
  * -- base_score：初始化打分
  * -- eval_metric：rmse、mae、logloss等 默认根据objective来选择合适的目标函数
  * 调参建议：
  * -- 调整max_depth、min_child_weight、gamma 减少模型复杂度
  * -- subsample、colsample_bytree
  * -- 调低eta，需要增加num_round
  * 数据不均衡如何解决：
  * -- 手动均衡数据集
  * -- 调整scale_pos_weight、max_delta_step参数
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
