package com.mtnfog;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.tokenize.WordpieceTokenizer;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class App {

    public static final String INPUT_IDS = "input_ids";
    public static final String ATTENTION_MASK = "attention_mask";
    public static final String TOKEN_TYPE_IDS = "token_type_ids";

    private static final String MODEL_PATH = "/home/jeff/Desktop/onnx-sentence-transformers/model/";
    public static void main(String[] args) throws OrtException, IOException {

        final String sentence = "george washington was president";
        //final String sentence = "Hello I'm a single sentence";

        final OrtEnvironment env = OrtEnvironment.getEnvironment();
        final OrtSession session = env.createSession(MODEL_PATH + "model.onnx", new OrtSession.SessionOptions());

        System.out.println("Loading vocab");
        final Map<String, Integer> vocab = loadVocab(new File(MODEL_PATH + "vocab.txt"));

        System.out.println("Creating tokenizer");
        final Tokenizer tokenizer = new WordpieceTokenizer(vocab.keySet());

        System.out.println("Tokenizing the input sentence");
        final List<Tokens> wordpieceTokens = tokenize(sentence, tokenizer, vocab);

        /*
        last_hidden_state - float32[x, x, x]
        924 - float32[Tanh924_dim_0, 384]   <----------- What we want.
        */

        for (final Tokens tokens : wordpieceTokens) {

            final Map<String, OnnxTensor> inputs = new HashMap<>();

            inputs.put(INPUT_IDS, OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.getIds()),
                new long[] {1, tokens.getIds().length}));

            inputs.put(ATTENTION_MASK, OnnxTensor.createTensor(env,
                LongBuffer.wrap(tokens.getMask()), new long[] {1, tokens.getMask().length}));

            inputs.put(TOKEN_TYPE_IDS, OnnxTensor.createTensor(env,
                LongBuffer.wrap(tokens.getTypes()), new long[] {1, tokens.getTypes().length}));

            System.out.println("Performing inference");
            final float[][][] v = (float[][][]) session.run(inputs).get(0).getValue();

            System.out.print("[[");

            for(int i = 0; i < v[0].length; i++) {

                int count = 1;

                for(int j = 0; j < v[0][0].length; j++) {

                    System.out.print(" " + v[0][0][j] + " ");

                    count++;

                    if(count == 5) {
                        count = 1;
                        System.out.print("\n");
                    }

                }

            }

            System.out.print("]]");

        }

    }

    private static Map<String, Integer> loadVocab(File vocab) throws IOException {

        final Map<String, Integer> v = new HashMap<>();

        try (final BufferedReader br = new BufferedReader(new FileReader(vocab.getPath()))) {

            String line = br.readLine();
            int x = 0;

            while (line != null) {

                line = br.readLine();
                x++;

                v.put(line, x);

            }

        }

        return v;

    }

    private static List<Tokens> tokenize(final String text, Tokenizer tokenizer, Map<String, Integer> vocab) {

        final List<Tokens> t = new LinkedList<>();

        // Now we can tokenize the group and continue.
        final String[] tokens = tokenizer.tokenize(text);

        final int[] ids = new int[tokens.length];
        //final int[] ids = new int[256];

        final long[] mask = new long[ids.length];

        for (int x = 0; x < tokens.length; x++) {
            ids[x] = vocab.get(tokens[x]);

            System.out.println(x + " = " + vocab.get(tokens[x]));

        //for (int x = 0; x < 256; x++) {

          /*  if(x < tokens.length) {
                ids[x] = vocab.get(tokens[x]);
                mask[x] = 1;
            } else {
                // This is padding.
                ids[x] = 0;
                mask[x] = 0;
            }*/

        }

        System.out.print("\n");

        final long[] lids = Arrays.stream(ids).mapToLong(i -> i).toArray();

        for(int x = 0; x < lids.length; x++) {
            System.out.print(lids[x] + " ");
        }

        final long[] types = new long[ids.length];
        Arrays.fill(types, 1);

        System.out.println("Number of tokens: " + lids.length);

        t.add(new Tokens(tokens, lids, mask, types));

        return t;

    }

}
