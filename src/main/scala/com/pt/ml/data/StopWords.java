package com.pt.ml.data;

import com.google.common.io.Files;

import java.io.File;
import java.nio.charset.Charset;
import java.util.HashSet;
import java.util.Set;

public class StopWords {
    private static final Set<String> stopWordsEn = new HashSet<String>();
    private static final Set<String> stopWordsZh = new HashSet<String>();

    static {
        try {
            stopWordsEn.addAll(Files.readLines(new File("src/main/resources/stopWordsEn.txt"), Charset.forName("utf-8")));
            stopWordsZh.addAll(Files.readLines(new File("src/main/resources/stopWordsZh.txt"), Charset.forName("utf-8")));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static boolean isEnStopWords(String word) {
        return stopWordsEn.contains(word);
    }

    public static boolean isZhStopWords(String word) {
        return stopWordsZh.contains(word);
    }
}
