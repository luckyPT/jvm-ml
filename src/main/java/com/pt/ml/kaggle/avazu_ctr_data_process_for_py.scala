package com.pt.ml.kaggle

import java.util.Random

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

/**
 * 数据处理，用于python 训练深度学习模型；
 * 可与：https://github.com/luckyPT/QuickMachineLearning配合使用
 */
object avazu_ctr_data_process_for_py {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Program Files\\winutils\\")
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("kaggle_avazu_ctr")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._
    val hour2Week = udf {
      (dateStr: String) => {
        import java.text.SimpleDateFormat
        import java.util.Calendar
        val sdf = new SimpleDateFormat("yyyyMMddHH")
        val c = Calendar.getInstance()
        c.setTime(sdf.parse("20" + dateStr))
        c.get(Calendar.DAY_OF_WEEK)
      }
    }

    val hour2hour = udf {
      (dateStr: String) => {
        dateStr.substring(6).toInt
      }
    }

    val trainData = spark.read.option("header", true)
      .parquet("dataset/avazu_ctr/train_data.snappy.parquet")
      .withColumn("day_of_week", hour2Week($"hour"))
      .withColumn("hourIndex", hour2hour($"hour"))
      .drop($"hour")

    val testData = spark.read.option("header", true)
      .parquet("dataset/avazu_ctr/train_data.snappy.parquet")
      .withColumn("day_of_week", hour2Week($"hour"))
      .withColumn("hourIndex", hour2hour($"hour"))
      .drop($"hour")

    val categoryCols = Array("C1", "banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category",
      "device_id", "device_model", "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "hourIndex", "day_of_week")
    val categoryColsIndexer = categoryCols.map(indexer)
    val dataProcessPipeline = new Pipeline()
      .setStages(categoryColsIndexer)

    val model = dataProcessPipeline.fit(trainData)
    val trainDataCsv = model.transform(trainData)
      .select($"click", $"C1Index", $"banner_posIndex", $"site_idIndex", $"site_domainIndex", $"site_categoryIndex", $"app_idIndex",
        $"app_domainIndex", $"app_categoryIndex", $"device_idIndex", $"device_modelIndex", $"device_typeIndex", $"device_conn_typeIndex",
        $"C14Index", $"C15Index", $"C16Index", $"C17Index", $"C18Index", $"C19Index", $"C20Index", $"C21Index", $"hourIndex", $"day_of_week")
      .persist()
    trainDataCsv.repartition(1)
      .write.option("header", true).csv("dataset/avazu_ctr/process/train_data")

    val testDataCsv = model.transform(testData)
      .select($"id", $"C1Index", $"banner_posIndex", $"site_idIndex", $"site_domainIndex", $"site_categoryIndex", $"app_idIndex",
        $"app_domainIndex", $"app_categoryIndex", $"device_idIndex", $"device_modelIndex", $"device_typeIndex", $"device_conn_typeIndex",
        $"C14Index", $"C15Index", $"C16Index", $"C17Index", $"C18Index", $"C19Index", $"C20Index", $"C21Index", $"hourIndex", $"day_of_week")
      .persist()

    testDataCsv.repartition(1)
      .write.option("header", true).csv("dataset/avazu_ctr/process/test_data")

    trainDataCsv.union(testDataCsv)
      .select(max($"C1Index".cast(IntegerType)), max($"banner_posIndex".cast(IntegerType)), max($"site_idIndex".cast(IntegerType)),
        max($"site_domainIndex".cast(IntegerType)), max($"site_categoryIndex".cast(IntegerType)), max($"app_idIndex".cast(IntegerType)),
        max($"app_domainIndex".cast(IntegerType)), max($"app_categoryIndex".cast(IntegerType)), max($"device_idIndex".cast(IntegerType)),
        max($"device_modelIndex".cast(IntegerType)), max($"device_typeIndex".cast(IntegerType)), max($"device_conn_typeIndex".cast(IntegerType)),
        max($"C14Index".cast(IntegerType)), max($"C15Index".cast(IntegerType)), max($"C16Index".cast(IntegerType)),
        max($"C17Index".cast(IntegerType)), max($"C18Index".cast(IntegerType)), max($"C19Index".cast(IntegerType)),
        max($"C20Index".cast(IntegerType)), max($"C21Index".cast(IntegerType)), max($"hourIndex".cast(IntegerType)),
        max($"day_of_week".cast(IntegerType)))
      .repartition(1).write.option("header", true).csv("dataset/avazu_ctr/process/data_max_index")

    spark.read.option("header", true)
      .csv("dataset/avazu_ctr/process/train_data")
      .filter {
        r =>
          r.getAs[String]("click") == "1" || new Random().nextInt(1000) < 200
      }
      .repartition(1)
      .write.option("header", true).csv("dataset/avazu_ctr/process/train_data_sample")
  }


  def indexer(str: String): StringIndexer = {
    new StringIndexer()
      .setInputCol(str)
      .setOutputCol(str + "Index")
      .setHandleInvalid("keep")
  }
}
