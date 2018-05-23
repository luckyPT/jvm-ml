package com.pt.ml.data;

import com.google.common.io.Files;

import java.io.File;
import java.nio.charset.Charset;
import java.util.HashSet;
import java.util.Set;

public class StopWords {
    private static final Set<String> stopWords = new HashSet<String>();

    static {
        try {
            stopWords.addAll(Files.readLines(new File("src/main/resources/stopWords.txt"), Charset.forName("utf-8")));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static boolean isStopWords(String word) {
        return stopWords.contains(word);
    }
}
