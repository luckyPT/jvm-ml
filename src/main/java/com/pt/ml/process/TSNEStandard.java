package com.pt.ml.process;

import com.jujutsu.tsne.TSneConfiguration;
import com.jujutsu.tsne.barneshut.BHTSne;
import com.jujutsu.tsne.barneshut.BarnesHutTSne;
import com.jujutsu.tsne.barneshut.ParallelBHTsne;
import com.jujutsu.utils.TSneUtils;
import com.pt.ml.visualization.Scatter;
import org.spark_project.jetty.util.ArrayQueue;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

/**
 * 一种保分布的降维方式，据说是最好的降维可视化的手段;
 */
public class TSNEStandard {
    public static void main(String[] args) throws Exception {
        TSNEStandard.tsneVisualization2D("dataset/MINIST/train.csv", 784,
                true, 1000, 100.0, 3000);

    }

    /**
     * 2维可视化
     *
     * @param inputFile 输入文件（CSV 第一行是字段描述）
     * @param featureNum 特征数量
     * @param parallel 是否并发
     * @param maxIter 最大迭代次数
     * @param perplexity 一定范围内，值越大效果越好
     * @param sampleCount 可视化样本数量
     */
    public static void tsneVisualization2D(String inputFile, int featureNum, boolean parallel, int maxIter, double perplexity, int sampleCount) {
        try {
            File file = new File(inputFile);
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String[] label = new String[sampleCount];
            double[][] data = new double[sampleCount][featureNum];
            for (int i = 0; i < sampleCount + 1; i++) {
                StringBuilder stringBuilder = new StringBuilder(reader.readLine());
                if (i != 0) {//第一行是标题头
                    String[] strs = stringBuilder.toString().split(",");
                    label[i - 1] = strs[0];
                    for (int j = 1; j < featureNum; j++) {
                        data[i - 1][j - 1] = Double.parseDouble(strs[j]);
                    }
                }
            }
            BarnesHutTSne tsne = parallel ? new ParallelBHTsne() : new BHTSne();
            TSneConfiguration config = TSneUtils.buildConfig(data,
                    2,
                    featureNum,
                    perplexity,
                    maxIter);
            double[][] Y = tsne.tsne(config);
            Scatter scatter = new Scatter();
            Map<String, List<double[]>> map = new HashMap<>();
            for (int i = 0; i < Y.length; i++) {
                List<double[]> array = map.get(label[i]);
                if (array != null) {
                    array.add(new double[] {Y[i][0], Y[i][1]});
                } else {
                    array = new LinkedList<>();
                    array.add(new double[] {Y[i][0], Y[i][1]});
                    map.put(label[i], array);
                }
            }
            final Queue<Color> colors = new ArrayQueue<>();
            colors.add(Color.BLUE);
            colors.add(Color.RED);
            colors.add(Color.BLACK);
            colors.add(Color.BLUE);
            colors.add(Color.YELLOW);
            colors.add(Color.GREEN);
            colors.add(Color.GRAY);
            colors.add(Color.MAGENTA);
            colors.add(Color.PINK);
            colors.add(Color.ORANGE);
            colors.add(Color.CYAN);
            colors.add(Color.DARK_GRAY);

            for (Map.Entry<String, List<double[]>> entry : map.entrySet()) {
                List<double[]> entryValue = entry.getValue();
                double x[] = new double[entryValue.size()];
                double y[] = new double[entryValue.size()];
                for (int i = 0; i < entryValue.size(); i++) {
                    x[i] = entryValue.get(i)[0];
                    y[i] = entryValue.get(i)[1];
                }
                scatter.addData(entry.getKey(), x, y, colors.poll());
            }
            scatter.showPlot();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
