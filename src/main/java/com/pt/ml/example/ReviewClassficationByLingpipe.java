package com.pt.ml.example;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.classify.DynamicLMClassifier;
import com.aliasi.classify.JointClassification;
import com.aliasi.classify.JointClassifier;
import com.aliasi.lm.NGramProcessLM;
import com.aliasi.util.AbstractExternalizable;
import org.apache.commons.io.IOUtils;

import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * n-gram 分类,效果很接近SVM
 * http://alias-i.com/lingpipe/demos/tutorial/classify/read-me.html
 *
 * @see com.aliasi.classify.LMClassifier
 */
public class ReviewClassficationByLingpipe {
    public static void main(String[] args) throws Exception {
        System.out.println("-----start-----");
        String[] CATEGORIES = new String[] {"__label__0", "__label__1"};
        DynamicLMClassifier<NGramProcessLM> classifier = DynamicLMClassifier.createNGramProcess(CATEGORIES, 4);

        List<String> trainText = IOUtils.readLines(new FileInputStream(new File("dataset/word2vec-nlp-tutorial/fastTextTrain.txt")));
        trainText.forEach(line -> {
            String[] labelText = line.split("\t");
            Classified<CharSequence> classified = new Classified<>(labelText[1], new Classification(labelText[0]));
            classifier.handle(classified);
        });

        @SuppressWarnings("unchecked")
        JointClassifier<CharSequence> compiledClassifier = (JointClassifier<CharSequence>) AbstractExternalizable.compile(classifier);

        List<String> testText = IOUtils.readLines(new FileInputStream(new File("dataset/word2vec-nlp-tutorial/fastTextTest.txt")));
        //test
        Map<String, Long> confusionMatrix = new HashMap<>();
        testText.forEach(line -> {
            String[] labelText = line.split("\t");
            JointClassification jc = compiledClassifier.classify(labelText[1]);
            String key = "label:" + labelText[0] + ";pre:" + jc.bestCategory();
            long value = confusionMatrix.getOrDefault(key, 0L);
            confusionMatrix.put(key, ++value);
        });
        confusionMatrix.forEach((key, value) -> System.out.println(key + "\t" + value));
    }
}
