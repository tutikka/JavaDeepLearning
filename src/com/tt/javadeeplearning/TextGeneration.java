package com.tt.javadeeplearning;

import com.tt.javadeeplearning.layer.Dense;
import com.tt.javadeeplearning.layer.Dropout;
import com.tt.javadeeplearning.layer.LSTM;
import com.tt.javadeeplearning.layer.Softmax;
import com.tt.javadeeplearning.loss.CategoricalCrossEntropy;
import com.tt.javadeeplearning.loss.SparseCategoricalCrossEntropy;
import com.tt.javadeeplearning.network.Network;

import java.util.*;

import static com.tt.javadeeplearning.postprocess.PostProcess.argmax;

public class TextGeneration {

    public static class TextProcessor {

        private Map<Character, Integer> charToIndex = new HashMap<>();

        private Map<Integer, Character> indexToChar = new HashMap<>();

        public TextProcessor(String text) {
            Set<Character> uniqueChars = new HashSet<>();
            for (char c : text.toCharArray()) {
                uniqueChars.add(c);
            }
            List<Character> sortedChars = new ArrayList<>(uniqueChars);
            Collections.sort(sortedChars);
            for (int i = 0; i < sortedChars.size(); i++) {
                charToIndex.put(sortedChars.get(i), i);
                indexToChar.put(i, sortedChars.get(i));
            }
        }

        public int[] textToSequence(String text) {
            int[] result = new int[text.length()];
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                result[i] = charToIndex.get(c);
            }
            return (result);
        }

        public String sequenceToText(int[] sequence) {
            StringBuilder sb = new StringBuilder();
            for (int idx : sequence) {
                sb.append(indexToChar.get(idx));
            }
            return sb.toString();
        }

        public Character intToText(int i) {
            return (indexToChar.get(i));
        }

        public int vocabularySize() {
            return charToIndex.size();
        }
    }

    public static void main(String[] args) {

        String text = "Once upon a time, in a faraway land, there was a brave knight.";
        TextProcessor textProcessor = new TextProcessor(text);
        System.out.println("Text length: " + text.length());
        System.out.println("Vocabulary size: " + textProcessor.vocabularySize());

        int sequenceLength = 10;

        Network network = new Network();
        network.addLayer(new LSTM(sequenceLength, 256));
        network.addLayer(new Dense(256, textProcessor.vocabularySize()));
        network.addLayer(new Softmax());

        int[] encodedText = textProcessor.textToSequence(text);
        double[][] features = new double[encodedText.length - sequenceLength][sequenceLength];
        double[] labels = new double[encodedText.length - sequenceLength];

        for (int i = 0; i < encodedText.length - sequenceLength; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                features[i][j] = encodedText[i + j];
            }
            labels[i] = encodedText[i + sequenceLength];  // Use the next character as the label
        }

        network.setTrain(true);
        network.train(features, labels, new SparseCategoricalCrossEntropy(), 5000, 0.005);
        network.setTrain(false);

        double[] input = new double[sequenceLength];
        int[] seed = textProcessor.textToSequence("there was");
        for (int i = 0; i < seed.length; i++) {
            input[i] = seed[i];
        }

        StringBuilder output = new StringBuilder("there was");
        for (int i = 0; i < 40; i++) {
            double[] prediction = network.predict(input);
            int nextToken = argmax(prediction);
            char nextChar = textProcessor.intToText(nextToken);
            output.append(nextChar);
            for (int j = 0; j < sequenceLength - 1; j++) {
                input[j] = input[j + 1];
            }
            input[sequenceLength - 1] = nextToken;
        }

        System.out.println(output.toString());

    }

}
