package com.pt.ml.deeplearning;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

/**
 * @see CnnNeuralNetwork
 */
public class DeeplearningUI {
    public static StatsListener startUI() {
        UIServer server = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        server.attach(statsStorage);
        return new StatsListener(statsStorage);
    }
}
