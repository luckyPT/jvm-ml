package com.pt.ml.process;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 一般开源的分词模型比较通用，所以需要无论是在词典还是算法上都相对复杂；
 * 工程中往往涉及到具体领域内的应用时，需要兼顾算法与工程性能，因此定义简易的分词模型
 * 牺牲一定的分词算法效果(甚至不牺牲效果)，提高工程可用性指标 包括：内存、初始化耗时、分词耗时等
 */
public class CustomSegmenter {
    private TrieTree tree;

    public CustomSegmenter(String[] words) {
        tree = new TrieTree(words);
    }

    public static void main(String[] args) {
        String[] testStrs = new String[]{
                "已发货", "已签收", "已签单", "已经签收", "已购", "红包", "礼券", "大额"
        };
        String s = "您已签收订单，5星好评领大额红包哦。";
        CustomSegmenter segmenter = new CustomSegmenter(testStrs);
        List<String> tokens = segmenter.process(s);
        tokens.forEach(w -> System.out.print(w + "||"));
    }

    public List<String> process(String s) {
        List<String> result = new ArrayList<>();
        for (int i = 0; i < s.length(); ) {
            int lastIndex = tree.maxLastIndex(s, i);
            result.add(s.substring(i, lastIndex));
            i = lastIndex;
        }
        return result;
    }

    static class TrieTree {
        Node root = new Node('\u0000');

        int maxLastIndex(String s, int startIndex) {
            Node curNode = root;
            for (int i = startIndex; i < s.length(); i++) {
                char c = s.charAt(i);
                curNode = curNode.toNextNode(c);
                if (curNode == null) {
                    return startIndex + 1;
                }
                if (curNode.isEnd()) {
                    return i + 1;
                }
            }
            return startIndex + 1;
        }

        TrieTree(String[] words) {
            for (String w : words) {
                Node curNode = root;
                for (char c : w.toCharArray()) {
                    if (curNode.toNextNode(c) == null) {
                        curNode.addNextNode(c);
                    }
                    curNode = curNode.toNextNode(c);
                }
            }
        }

        static class Node {
            char c;
            Map<Character, Node> nodes;

            Node(char c) {
                this.c = c;
            }

            Node toNextNode(char c) {
                return nodes == null ? null : nodes.get(c);
            }

            void addNextNode(char c) {
                if (nodes == null) {
                    nodes = new HashMap<>();
                }
                nodes.put(c, new Node(c));
            }

            boolean isEnd() {
                return nodes == null;
            }
        }
    }
}
