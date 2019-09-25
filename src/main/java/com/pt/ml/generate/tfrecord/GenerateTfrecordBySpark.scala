package com.pt.ml.generate.tfrecord

import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._

object GenerateTfrecordBySpark {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[4]")
                .appName("HelloWorld")
                .getOrCreate()

        val testRows: Array[Row] = Array(
            new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, List(1.0, 2.0), "r1")),
            new GenericRow(Array[Any](21, 2, 24L, 12.0F, 15.0, List(2.0, 2.0), "r2")))
        val schema = StructType(List(StructField("id", IntegerType),
            StructField("IntegerCol", IntegerType),
            StructField("LongCol", LongType),
            StructField("FloatCol", FloatType),
            StructField("DoubleCol", DoubleType),
            StructField("VectorCol", ArrayType(DoubleType, true)),
            StructField("StringCol", StringType)))

        val rdd = spark.sparkContext.parallelize(testRows)

        val df: DataFrame = spark.createDataFrame(rdd, schema)
        df.repartition(1).write.format("tfrecords").option("recordType", "Example").save("./tfrecord")
    }
}

/** 对应python读取代码如下
  *
  *
import tensorflow as tf

tf.enable_eager_execution()
feature_description = {
    'id': tf.io.FixedLenFeature([], tf.int64),
    'IntegerCol': tf.io.FixedLenFeature([], tf.int64),
    'LongCol': tf.io.FixedLenFeature([], tf.int64),
    'FloatCol': tf.io.FixedLenFeature([], tf.float32),
    'DoubleCol': tf.io.FixedLenFeature([], tf.float32),
    'VectorCol': tf.io.FixedLenFeature([2], tf.float32),
    'StringCol': tf.io.FixedLenFeature([], tf.string),
}


def my_print(x):
    for i in x:
        print('--', i, x[i])
    return x


def parse_fn(example_proto):
    # 返回一个dic 特征到特征值的映射
    return tf.io.parse_single_example(example_proto, feature_description)


ds = tf.data.TFRecordDataset("/home/panteng/IdeaProjects/jvm-ml/tfrecord/part-r-00000")
ds = ds.map(parse_fn).map(my_print)

for parsed_record in ds.take(2):  # 或者for parsed_record in ds.take(10)
    print(parsed_record['id'],
          parsed_record['IntegerCol'],
          parsed_record['LongCol'],
          parsed_record['FloatCol'],
          parsed_record['DoubleCol'],
          parsed_record['VectorCol'],
          parsed_record['StringCol'].numpy().decode(encoding='utf-8'))

*/
