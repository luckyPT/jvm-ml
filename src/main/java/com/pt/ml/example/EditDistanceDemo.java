package com.pt.ml.example;

import com.aliasi.spell.EditDistance;
import com.aliasi.util.Distance;

/**
 * 编辑距离的计算：
 * http://alias-i.com/lingpipe/docs/api/index.html
 * 逐个字符对比，通过 Insert\Delete\Substitute\Transpose（插入、删除、替换、交换） 操作使得每个字符保持一致，每个操作加1，最后求和
 * Levenshtein distance计算方式不允许使用Transpose操作
 * Damerau-Levenstein计算方式允许使用Transpose操作（仅仅是相邻的交换）
 * 对于abc和acb 允许交换距离为1,不允许距离为2
 */
public class EditDistanceDemo {
    public static void main(String[] args) {
        //false - Levenshtein distance; true - Damerau-Levenstein distance
        Distance<CharSequence> EDIT_DISTANCE = new EditDistance(false);
        System.out.println(EDIT_DISTANCE.distance("我早上吃饭了","你早上吃的饭了吗"));
    }
}
