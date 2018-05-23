package com.pt.ml.process;

import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;
import com.pt.ml.data.StopWords;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 仅支持分词
 */
public class JiebaSegmenterUtil {
    private static final JiebaSegmenter segmenter = new JiebaSegmenter();

    public static List<String> getWords(String text) {
        List<SegToken> terms = segmenter.process(text, JiebaSegmenter.SegMode.SEARCH);
        List<String> words = new ArrayList<>();
        for (SegToken term : terms) {
            String word = term.word;
            if (StringUtils.isNotBlank(word)) {
                words.add(word);
            }
        }
        return words;
    }

    public static List<String> getWordsRemovedStopWords(String text) {
        List<SegToken> terms = segmenter.process(text, JiebaSegmenter.SegMode.SEARCH);
        List<String> words = new ArrayList<>();
        for (SegToken term : terms) {
            String word = term.word;
            if (StringUtils.isNotBlank(word) && !StopWords.isStopWords(word)) {
                words.add(word);
            }
        }
        return words;
    }

    public static void main(String[] args) {
        getWords("大家早上好，京东发优惠券了啊！").forEach(str -> System.out.print(str + " "));
        System.out.println();
        getWordsRemovedStopWords("大家早上好，京东发优惠券了啊！").forEach(str -> System.out.print(str + " "));
    }
}
