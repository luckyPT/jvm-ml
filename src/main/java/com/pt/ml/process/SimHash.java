package com.pt.ml.process;

import java.util.Arrays;
import java.util.List;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;

/**
 * 用于估计两个集合的相似程度，具体算法如下：
 * 1.首先设定hash值的位数（比如8位）
 * 2.计算集合中每一个元素的hash值，然后转化为二进制
 * 3.将元素的哈希值按位求和（求和规则，如果是0则加-1,是1则加1）；
 * 比如元素1的hash值为：00100101,
 * 元素2的哈希值为：01010101；
 * 则求和之后为：-2/0/0/0/-2/2/-2/2；然后如果这个位>0 则为1,否则为0
 * 0 0 0 0 0  1  0 1
 * 4.最终的hash值为00000101
 *
 * 这种算法可用于比较两个短文本的（长文本一定程度上也可行，比如网页去重）的相似度；短文本可以看成是词语的集合
 *
 * 其他类似算法：
 * - 最短编辑路径
 * - 基于词袋
 * - 余弦相似
 * 等
 * 本代码参考：https://github.com/sing1ee/simhash-java
 */
public class SimHash {
    public static void main(String[] args) {
        String[] strs = new String[] {
                "the cat sat on the mat",
                "the cat sat on a mat",
                "we all scream for ice cream"
        };
        SimHash simHash = new SimHash();
        long[] hashes = new long[strs.length];
        for (int i = 0; i < strs.length; i++) {
            hashes[i] = simHash.simhash64(strs[i]);
        }
        System.out.println(simHash.hammingDistance(hashes[0], hashes[1]));
        System.out.println(simHash.hammingDistance(hashes[0], hashes[2]));
        System.out.println(simHash.hammingDistance(hashes[1], hashes[2]));
    }

    public long simhash64(String doc) {
        int bitLen = 64;
        int[] bits = new int[bitLen];
        List<String> tokens = Arrays.asList(doc.split(""));
        for (String t : tokens) {
            long v = MurmurHash.hash64(t);
            for (int i = bitLen; i >= 1; --i) {
                if (((v >> (bitLen - i)) & 1) == 1) {
                    ++bits[i - 1];
                } else {
                    --bits[i - 1];
                }
            }
        }
        long hash = 0x0000000000000000;
        long one = 0x0000000000000001;
        for (int i = bitLen; i >= 1; --i) {
            if (bits[i - 1] > 0) {
                hash |= one;
            }
            one = one << 1;
        }
        return hash;
    }

    public long simhash32(String doc) {
        int bitLen = 32;
        int[] bits = new int[bitLen];
        List<String> tokens = Arrays.asList(doc.split(""));
        for (String t : tokens) {
            int v = MurmurHash.hash32(t);
            for (int i = bitLen; i >= 1; --i) {
                if (((v >> (bitLen - i)) & 1) == 1) {
                    ++bits[i - 1];
                } else {
                    --bits[i - 1];
                }
            }
        }
        int hash = 0x00000000;
        int one = 0x00000001;
        for (int i = bitLen; i >= 1; --i) {
            if (bits[i - 1] > 1) {
                hash |= one;
            }
            one = one << 1;
        }
        return hash;
    }

    public int hammingDistance(long hash1, long hash2) {
        long i = hash1 ^ hash2;
        i = i - ((i >>> 1) & 0x5555555555555555L);
        i = (i & 0x3333333333333333L) + ((i >>> 2) & 0x3333333333333333L);
        i = (i + (i >>> 4)) & 0x0f0f0f0f0f0f0f0fL;
        i = i + (i >>> 8);
        i = i + (i >>> 16);
        i = i + (i >>> 32);
        return (int) i & 0x7f;
    }

    static class MurmurHash {
        static int hash32(String doc) {
            byte[] buffer = doc.getBytes(Charset.forName("utf-8"));
            ByteBuffer data = ByteBuffer.wrap(buffer);
            return hash32(data, 0, buffer.length, 0);
        }

        static int hash32(ByteBuffer data, int offset, int length, int seed) {
            int m = 0x5bd1e995;
            int r = 24;

            int h = seed ^ length;

            int len_4 = length >> 2;

            for (int i = 0; i < len_4; i++) {
                int i_4 = i << 2;
                int k = data.get(offset + i_4 + 3);
                k = k << 8;
                k = k | (data.get(offset + i_4 + 2) & 0xff);
                k = k << 8;
                k = k | (data.get(offset + i_4 + 1) & 0xff);
                k = k << 8;
                k = k | (data.get(offset + i_4) & 0xff);
                k *= m;
                k ^= k >>> r;
                k *= m;
                h *= m;
                h ^= k;
            }

            // avoid calculating modulo
            int len_m = len_4 << 2;
            int left = length - len_m;

            if (left != 0) {
                if (left >= 3) {
                    h ^= (int) data.get(offset + length - 3) << 16;
                }
                if (left >= 2) {
                    h ^= (int) data.get(offset + length - 2) << 8;
                }
                if (left >= 1) {
                    h ^= (int) data.get(offset + length - 1);
                }

                h *= m;
            }

            h ^= h >>> 13;
            h *= m;
            h ^= h >>> 15;

            return h;
        }

        static long hash64(String doc) {
            byte[] buffer = doc.getBytes(Charset.forName("utf-8"));
            ByteBuffer data = ByteBuffer.wrap(buffer);
            return hash64(data, 0, buffer.length, 0);
        }

        static long hash64(ByteBuffer key, int offset, int length, long seed) {
            long m64 = 0xc6a4a7935bd1e995L;
            int r64 = 47;

            long h64 = (seed & 0xffffffffL) ^ (m64 * length);

            int lenLongs = length >> 3;

            for (int i = 0; i < lenLongs; ++i) {
                int i_8 = i << 3;

                long k64 = ((long) key.get(offset + i_8 + 0) & 0xff)
                        + (((long) key.get(offset + i_8 + 1) & 0xff) << 8)
                        + (((long) key.get(offset + i_8 + 2) & 0xff) << 16)
                        + (((long) key.get(offset + i_8 + 3) & 0xff) << 24)
                        + (((long) key.get(offset + i_8 + 4) & 0xff) << 32)
                        + (((long) key.get(offset + i_8 + 5) & 0xff) << 40)
                        + (((long) key.get(offset + i_8 + 6) & 0xff) << 48)
                        + (((long) key.get(offset + i_8 + 7) & 0xff) << 56);

                k64 *= m64;
                k64 ^= k64 >>> r64;
                k64 *= m64;

                h64 ^= k64;
                h64 *= m64;
            }

            int rem = length & 0x7;

            switch (rem) {
                case 0:
                    break;
                case 7:
                    h64 ^= (long) key.get(offset + length - rem + 6) << 48;
                case 6:
                    h64 ^= (long) key.get(offset + length - rem + 5) << 40;
                case 5:
                    h64 ^= (long) key.get(offset + length - rem + 4) << 32;
                case 4:
                    h64 ^= (long) key.get(offset + length - rem + 3) << 24;
                case 3:
                    h64 ^= (long) key.get(offset + length - rem + 2) << 16;
                case 2:
                    h64 ^= (long) key.get(offset + length - rem + 1) << 8;
                case 1:
                    h64 ^= (long) key.get(offset + length - rem);
                    h64 *= m64;
            }

            h64 ^= h64 >>> r64;
            h64 *= m64;
            h64 ^= h64 >>> r64;

            return h64;
        }
    }
}
