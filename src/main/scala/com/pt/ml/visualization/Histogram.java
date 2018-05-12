package com.pt.ml.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;
import javax.swing.*;

public class Histogram extends ApplicationFrame {
    private DefaultCategoryDataset categoryDataSet = new DefaultCategoryDataset();

    public Histogram() {
        this("应用名称");
    }

    public Histogram(String applicationTitle) {
        this(applicationTitle, "标题名称");
    }

    public Histogram(String applicationTitle, String chartTitle) {
        this(applicationTitle, chartTitle, "xLabel", "yLabel");
    }

    public Histogram(String applicationTitle, String chartTitle, String xLabel, String yLabel) {
        super(applicationTitle);
        JFreeChart jfreechart = ChartFactory.createBarChart(chartTitle, xLabel, yLabel, categoryDataSet, PlotOrientation.VERTICAL, true, true, false);
        JPanel jpanel = new ChartPanel(jfreechart);
        jpanel.setPreferredSize(new Dimension(550, 270));
        setContentPane(jpanel);
    }

    public void setDatasets(String[] category, double[] values) {
        setDatasets("默认", category, values);
    }

    public void setDatasets(String rowName, String[] category, double[] values) {
        assert category.length == values.length : "required category.length == values.length";
        for (int i = 0; i < category.length; i++) {
            categoryDataSet.addValue(values[i], rowName, category[i]);
        }
    }

    public void showPlot() {
        this.pack();
        RefineryUtilities.centerFrameOnScreen(this);
        this.setVisible(true);
    }

    public static void main(String args[]) {
        String[] x = new String[] {"A", "B", "C"};
        double[] y = new double[] {12, 23, 4};
        Histogram histogrami = new Histogram("应用名称");
        histogrami.setDatasets(x, y);
        histogrami.setDatasets("class2", new String[] {"A", "B", "C"}, new double[] {23, 26, 28});
        histogrami.showPlot();
    }
}