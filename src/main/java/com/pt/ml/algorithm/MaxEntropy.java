package com.pt.ml.algorithm;

import opennlp.tools.ml.AbstractTrainer;
import opennlp.tools.ml.EventTrainer;
import opennlp.tools.ml.TrainerFactory;
import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.maxent.GISTrainer;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.ml.model.MutableContext;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.TrainingParameters;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 理论知识：
 * 熵、联合熵、条件熵的定义；条件熵 = 熵 - 联合熵
 * 最大化的熵形式上看跟条件熵非常像(具体是不是说不太好)；
 * 约束条件：
 * 1.条件函数关于p~(x)、p(y|x)的期望 等于 条件函数关于p~(x,y)的期望(p~(x,y)是根据训练数据统计的，p(y|x)是待求解的)
 * 2.Σp(y|x) = 1
 * 更进一步：P(y|x) = softmax(Σw_i * f(x,y)_i)
 */
public class MaxEntropy {
    private static String[][] cntx = new String[][] {
            {"dog", "cat", "mouse"},
            {"text", "print", "mouse"},
            {"dog", "pig", "cat", "mouse"}
    };
    private static String[] outputs = new String[] {"A", "B", "A"};

    private static ObjectStream<Event> createEventStream() {
        List<Event> events = new ArrayList<>();
        for (int i = 0; i < cntx.length; i++) {
            events.add(new Event(outputs[i], cntx[i]));
        }
        return ObjectStreamUtils.createObjectStream(events);
    }

    public static void main(String[] args) throws IOException {
        TrainingParameters trainParams = new TrainingParameters();
        trainParams.put(AbstractTrainer.ALGORITHM_PARAM, GISTrainer.MAXENT_VALUE);
        trainParams.put(AbstractTrainer.CUTOFF_PARAM, 1);
        trainParams.put(GISTrainer.LOG_LIKELIHOOD_THRESHOLD_PARAM, 5.);

        EventTrainer trainer = TrainerFactory.getEventTrainer(trainParams, null);
        MaxentModel model = trainer.train(createEventStream());

        String[] context = {"cat", "pig", "mouse"};
        double[] result = model.eval(context);
        for (double r : result) {
            System.out.println(r);
        }
        CustomMaxentModel customMaxentModel = new CustomMaxentModel((GISModel) model);
        System.out.println(Arrays.toString(customMaxentModel.predict(context)));
    }

    static class CustomMaxentModel {
        Map<String, Double[]> params = new HashMap<>();
        int labelCount;

        CustomMaxentModel(AbstractModel model) {
            try {
                labelCount = model.getNumOutcomes();
                Field pmap = AbstractModel.class.getDeclaredField("pmap");
                pmap.setAccessible(true);
                HashMap<String, HashMap<String, MutableContext>> map = (HashMap<String, HashMap<String, MutableContext>>) pmap.get(model);

                Field outcoms = Context.class.getDeclaredField("outcomes");
                Field parameters = Context.class.getDeclaredField("parameters");
                outcoms.setAccessible(true);
                parameters.setAccessible(true);

                for (Map.Entry<String, HashMap<String, MutableContext>> entry : map.entrySet()) {
                    Double[] weights = new Double[labelCount];
                    int[] outcomsId = (int[]) outcoms.get(entry.getValue());
                    double[] modelWeights = (double[]) parameters.get(entry.getValue());
                    for (int i = 0; i < outcomsId.length; i++) {
                        weights[outcomsId[i]] = modelWeights[i];
                    }
                    params.put(entry.getKey(), weights);
                }
            } catch (NoSuchFieldException | IllegalAccessException e) {
                e.printStackTrace();
            }
        }

        double[] predict(String... args) {
            double[] ret = new double[labelCount];
            for (String f : args) {
                Double[] weight = params.get(f);
                for (int i = 0; i < weight.length; i++) {
                    if (weight[i] != null) {
                        ret[i] += weight[i];
                    }
                }
            }
            double sum = 0;
            for (int i = 0; i < labelCount; i++) {
                ret[i] = Math.exp(ret[i]);
                sum += ret[i];
            }
            for (int i = 0; i < labelCount; i++) {
                ret[i] = ret[i] / sum;
            }
            return ret;
        }
    }
}
