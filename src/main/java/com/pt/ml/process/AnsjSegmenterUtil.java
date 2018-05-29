package com.pt.ml.process;

import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;
import com.pt.ml.data.StopWords;
import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 支持分词和词性标注
 */
public class AnsjSegmenterUtil {

    public static List<String> getWords(String text) {
        Result terms = ToAnalysis.parse(text);
        List<String> words = new ArrayList<>();
        for (Term term : terms) {
            String word = term.getName();
            if (StringUtils.isNotBlank(word)) {
                words.add(word);
            }
        }
        return words;
    }

    public static List<String> getWordsRemovedStopWords(String text) {
        Result terms = ToAnalysis.parse(text);
        List<String> words = new ArrayList<>();
        for (Term term : terms) {
            String word = term.getName();
            if (StringUtils.isNotBlank(word) && !StopWords.isZhStopWords(word)) {
                words.add(word);
            }
        }
        return words;
    }

    public static void main(String[] args) {
        getWords("大家早上好，京东发优惠券了啊！").forEach(str -> System.out.print(str + "/"));
        System.out.println();
        getWordsRemovedStopWords("大家早上好，京东发优惠券了啊！").forEach(str -> System.out.print(str + "/"));
    }
}
