package com.pt.ml.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;

public class Line extends ApplicationFrame {
    private final XYSeriesCollection dataSet = new XYSeriesCollection();
    private ChartPanel chartPanel;
    private int lineCount = 0;
    private XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

    public Line() {
        this("应用名称");
    }

    public Line(String applicationTitle) {
        this(applicationTitle, "图表标题");
    }

    public Line(String applicationTitle, String chartTitle) {
        this(applicationTitle, chartTitle, "xLabel", "yLabel");
    }

    public Line(String applicationTitle, String chartTitle, String xLabel, String yLabel) {
        super(applicationTitle);
        JFreeChart lineChart = ChartFactory.createXYLineChart(
                chartTitle,
                xLabel,
                yLabel,
                dataSet,
                PlotOrientation.VERTICAL,
                true, true, false);
        chartPanel = new ChartPanel(lineChart);
        chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
        lineChart.getXYPlot().setRenderer(renderer);
    }

    public void linePlot(String name, double[] x, double[] y, Color color, float width) {
        assert x.length == y.length : "require x.length == y.length ";

        final XYSeries newLine = new XYSeries(name);
        for (int i = 0; i < x.length; i++) {
            newLine.add(x[i], y[i]);
        }
        dataSet.addSeries(newLine);

        this.renderer.setSeriesPaint(lineCount, color);
        this.renderer.setSeriesStroke(lineCount, new BasicStroke(width));
        lineCount++;

    }

    public void linePlot(String name, double[] x, double[] y) {
        linePlot(name, x, y, Color.RED, 2);
    }

    public void showPlot() {
        setContentPane(chartPanel);
        this.pack();
        RefineryUtilities.centerFrameOnScreen(this);
        this.setVisible(true);
    }

    public static void main(String[] args) {
        double[] x = new double[20];
        double[] y = new double[20];
        for (int i = 0; i < 20; i++) {
            x[i] = i * 0.5;
            y[i] = Math.sin(i * 0.5);
        }

        Line chart = new Line("应用名称", "图表标题", "X轴", "Y轴");
        chart.linePlot("红线", new double[] {1.0, 2.0, 3.0}, new double[] {1.0, 2.0, 3.0});
        chart.linePlot("蓝线", new double[] {2.0, 3.0, 4.0}, new double[] {3.0, 4.0, 5.0}, Color.BLUE, 2);
        chart.linePlot("黑线", new double[] {0.6, 0.8, 1.0}, new double[] {2.0, 1.6, 1.2}, Color.BLACK, 2);
        chart.linePlot("sin", x, y, Color.GREEN, 3);
        chart.showPlot();
    }
}