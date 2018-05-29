package com.pt.ml.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;

public class Scatter extends ApplicationFrame {
    private DefaultXYDataset dataSet = new DefaultXYDataset();
    private XYDotRenderer renderer = new XYDotRenderer();
    private ChartPanel chartPanel;
    private int categoryCount = 0;

    public Scatter() {
        this("应用名称");
    }

    public Scatter(String applicationTitle) {
        this(applicationTitle, "图表标题");
    }

    public Scatter(String applicationTitle, String chartTitle) {
        this(applicationTitle, chartTitle, "xLabel", "yLabel");
    }

    public Scatter(String applicationTitle, String chartTitle, String xLabel, String yLabel) {
        super(applicationTitle);
        renderer.setDotWidth(6);
        renderer.setDotHeight(6);

        JFreeChart chart = ChartFactory.createScatterPlot(chartTitle, xLabel, yLabel, dataSet, PlotOrientation.VERTICAL, true, false, false);
        chart.getXYPlot().setRenderer(renderer);
        chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
    }

    public void addData(String catagoryName, double[] x, double[] y, Color color) {
        assert x.length == y.length : "require x.length == y.length ";
        double[][] data = new double[2][x.length];
        for (int i = 0; i < x.length; i++) {
            data[0][i] = x[i];
            data[1][i] = y[i];
        }
        dataSet.addSeries(catagoryName, data);
        renderer.setSeriesPaint(categoryCount, color);
        categoryCount++;
    }

    public void showPlot() {
        setContentPane(chartPanel);
        this.pack();
        RefineryUtilities.centerFrameOnScreen(this);
        this.setVisible(true);
    }

    public static void main(String[] args) {
        // TODO Auto-generated method stub  
        double[] x = {15.0, 16.0, 19.0};
        double[] y = {12.0, 13.0, 14.0};
        Scatter scatter = new Scatter();
        scatter.addData("绿色", x, y, Color.GREEN);
        scatter.addData("蓝色", new double[] {9.0, 7.0, 5.0}, new double[] {4.0, 6.0, 8.0}, Color.BLUE);
        scatter.showPlot();
    }

}  