package com.pt.ml.deeplearning;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

public class File2DataSetIterator {
    /**
     * svm 将svm文件转为可训练的数据
     *
     * @param filePath 文件路径
     * @param featureNum 特征维度
     * @param labelNum 类别数
     * @param batchSize 批量大小
     * @return
     * @throws Exception
     */
    public static DataSetIterator svm(String filePath, int featureNum, int labelNum, int batchSize) throws Exception {
        File file = new File(filePath);
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_LABEL_INDEXING, true);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, featureNum);
        SVMLightRecordReader svmLightRecordReader = new SVMLightRecordReader();
        svmLightRecordReader.initialize(config, new FileSplit(file));
        return new RecordReaderDataSetIterator(svmLightRecordReader, batchSize, featureNum, labelNum);
    }

    public static DataSetIterator csv(String filePath, int featureNum, int labelNum, int batchSize,
                                      int skipLinesCount) throws Exception {
        File file = new File(filePath);
        CSVRecordReader csvRecordReader = new CSVRecordReader(skipLinesCount, ',');
        csvRecordReader.initialize(new FileSplit(file));
        return new RecordReaderDataSetIterator(csvRecordReader, batchSize, 0, labelNum);
    }

}
