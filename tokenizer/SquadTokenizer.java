
import static java.lang.Math.toIntExact;

import java.io.IOException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.StringReader;

import java.util.Arrays;
import java.util.List;
import java.util.LinkedList;
import java.util.Set;
import java.util.HashSet;
import java.util.Collections;
import java.util.Iterator;
import java.util.Properties;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

// http://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/json-simple/json-simple-1.1.1.jar
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

// http://argparse4j.github.io/
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.Namespace;

// https://stanfordnlp.github.io/CoreNLP/index.html
// http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/process/PTBTokenizer.html
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordToSentenceProcessor;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.pipeline.Annotation;


public class SquadTokenizer {

  private static Charset CHARSET = StandardCharsets.UTF_8;
  private static int MAX_SENTENCE_LENGTH = 150;

  private static class TokenizedText {
    List<String> tokens = new LinkedList<String>();
    List<String> originals = new LinkedList<String>();
    List<String> whitespaceAfters = new LinkedList<String>();
    List<Integer> startCharIdxs = new LinkedList<Integer>();
    List<Integer> afterEndCharIdxs = new LinkedList<Integer>();
    List<String> posTags = new LinkedList<String>();
    List<String> nerTags = new LinkedList<String>();
    List<Integer> tokenSentenceIdxs = new LinkedList<Integer>();
    List<Integer> sentenceLengths = new LinkedList<Integer>();
  }

  private static class JsonMetadata {
    Set<String> allTokens = new HashSet<String>();
    Set<String> unknownTokens = new HashSet<String>();
    int numArticles = 0;
    int numInvalidArticles = 0;
    int numParagraphs = 0;
    int numInvalidParagraphs = 0;
    int numQuestions = 0;
    int numFullQuestions = 0;
    int numPartialQuestions = 0;
    int numInvalidQuestions = 0;
    int numAnswers = 0;
    int numInvalidAnswersAlign = 0;     // failed matching answer string to word tokens
    int numInvalidAnswersSentence = 0;  // answer spans more than a single sentence
    int numSplits = 0;                  // hyphenated token split into constituent tokens
    int numSentenceJoins = 0;           // number of fixed / joined sentences
    int numSentenceBreaks = 0;          // number of long sentences that were arbitrarily split
    @Override
    public String toString() {
      return "word-types: " + allTokens.size() +
        "\nunknown word-types: " + unknownTokens.size() +
        "\n\t(invalidity is due to invalidity of answers):" +
        "\narticles: " + numArticles + " (invalid: " + numInvalidArticles + ")" +
        "\nparagraphs: " + numParagraphs + " (invalid: " + numInvalidParagraphs + ")" +
        "\nquestions: " + numQuestions + " (invalid: " + numInvalidQuestions +
        ", some answers: " + numPartialQuestions + ", all answers: " + numFullQuestions + ")" +
        "\nanswers: " + numAnswers + " (invalid align: " + numInvalidAnswersAlign + ", invalid sentence: " + numInvalidAnswersSentence + ")" +
        "\nnum of performed splits: " + numSplits +
        "\nnum of joined sentences: " + numSentenceJoins +
        "\nnum of broken sentences: " + numSentenceBreaks;
    }
  }


  public static void main(String[] args) throws IOException, ParseException {
    ArgumentParser parser = ArgumentParsers.newArgumentParser("SquadTokenizer").defaultHelp(true);
    parser.addArgument("in_json").help("input JSON file");
    parser.addArgument("out_json").help("output JSON file");
    parser.addArgument("--words_txt").required(true).help(
      "text file listing all known words (from GloVe)");
    parser.addArgument("--has_answers").action(Arguments.storeTrue()).help(
      "whether input JSON contains answers (as in train / dev set)");
    parser.addArgument("--split").action(Arguments.storeFalse()).help(
      "whether to split hyphenated words, when constituent tokens are found in GloVe - Enabled by default");
    parser.addArgument("--annotate").action(Arguments.storeTrue()).help(
      "whether to annotate with POS and with NER tags - This is currently unused");
    parser.addArgument("--max_sentence_length").type(Integer.class).setDefault(MAX_SENTENCE_LENGTH).help(
      "maximal sentence length, longer sentences arbitrarily split");
    parser.addArgument("--verbose").action(Arguments.storeTrue());

    Namespace ns = parser.parseArgsOrFail(args);
    System.out.println("\nSquadTokenizer.java : arguments : " + ns);

    String inJson = ns.getString("in_json");
    String outJson = ns.getString("out_json");
    boolean hasAnswers = ns.getBoolean("has_answers");
    String wordsTxt = ns.getString("words_txt");
    boolean split = ns.getBoolean("split");
    boolean annotate = ns.getBoolean("annotate");
    int maxSentenceLength = ns.getInt("max_sentence_length");
    boolean verbose = ns.getBoolean("verbose");

    System.out.println("Reading known words from " + wordsTxt);
    Set<String> knownWords = readKnownWords(wordsTxt);

    System.out.println("Initializing annotation pipeline");
    Properties props = new Properties();
    props.setProperty("annotators",  "tokenize, ssplit, pos, lemma, ner");
    String tokOpts = "invertible=true,untokenizable=" + (verbose ? "allKeep" : "noneKeep") + ",normalizeParentheses=false,normalizeOtherBrackets=false";
    props.setProperty("tokenize.options", tokOpts);
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    System.out.println("Writing tokenized version of " + inJson + " to " + outJson);
    JsonMetadata jsonMetadata = writeTokenizedJson(inJson, outJson, pipeline, knownWords, hasAnswers, split, annotate, maxSentenceLength, verbose);
    System.out.println(jsonMetadata + "\n");
  }


  private static Set<String> readKnownWords(String inFilename) throws IOException {
    Set<String> knownWords= new HashSet<String>();
    try (
      BufferedReader fin = new BufferedReader(new InputStreamReader(new FileInputStream(inFilename), CHARSET));
    ) {
      for(String line; (line = fin.readLine()) != null; ) {
        knownWords.add(line); // readLine removes EOL chars
      }
    }
    System.out.println("Number of known words: " + knownWords.size() + "\n");
    return knownWords;
  }


  private static JsonMetadata writeTokenizedJson(
    String inFilename, String outFilename, StanfordCoreNLP pipeline, Set<String> knownWords,
      boolean hasAnswers, boolean split, boolean annotate, int maxSentenceLength, boolean verbose)
      throws IOException, ParseException {
    try (
      BufferedReader fin = new BufferedReader(new InputStreamReader(new FileInputStream(inFilename), CHARSET));
      BufferedWriter fout = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outFilename), CHARSET))
    ) {
      JSONObject json = (JSONObject) new JSONParser().parse(fin);
      JsonMetadata jsonMetadata = addTokenData(json, pipeline, knownWords, hasAnswers, split, annotate, maxSentenceLength, verbose);
      fout.write(json.toJSONString());
      return jsonMetadata;
    }
  }


  @SuppressWarnings("unchecked")
  private static JsonMetadata addTokenData(
    JSONObject json, StanfordCoreNLP pipeline, Set<String> knownWords,
    boolean hasAnswers, boolean split, boolean annotate, int maxSentenceLength, boolean verbose) {
    JsonMetadata jsonMetadata = new JsonMetadata();

    Set<String> unknownWords = new HashSet<String>();
    JSONArray articleObjs = (JSONArray) json.get("data");
    if (articleObjs.isEmpty()) {
      throw new RuntimeException("No articles");
    }

    int numArticles = articleObjs.size();
    int numInvalidArticles = 0;

    int logNumQuestions = 0;

    for (Object articleObj : articleObjs) {
      JSONObject articleJson = (JSONObject) articleObj;
      String title = (String) articleJson.get("title");
      JSONArray paragraphObjs = (JSONArray) articleJson.get("paragraphs");
      if (paragraphObjs.isEmpty()) {
        throw new RuntimeException("No paragraphs\narticle:\n" + title);
      }

      int numParagraphs = paragraphObjs.size();
      int numInvalidParagraphs = 0;
      for (Object paragraphObj : paragraphObjs) {

        JSONObject paragraphJson = (JSONObject) paragraphObj;
        String contextStr = (String) paragraphJson.get("context");
        TokenizedText contextTok = tokenize(contextStr, pipeline, knownWords, unknownWords, split, maxSentenceLength, jsonMetadata, verbose);
        assertReconstruction("context", contextStr, contextTok, -1, -1);
        putTokens(paragraphJson, contextTok, annotate);
        jsonMetadata.allTokens.addAll(contextTok.tokens);
        JSONArray qaObjs = (JSONArray) paragraphJson.get("qas");
        if (qaObjs.isEmpty()) {
          throw new RuntimeException("No questions\narticle:\n" + title + "\ncontext:\n" + contextStr);
        }

        int numQuestions = qaObjs.size();
        int numFullQuestions = 0;
        int numPartialQuestions = 0;
        int numInvalidQuestions = 0;
        for (Object qaObj : qaObjs) {
          logNumQuestions += 1;
          if (logNumQuestions % 1000 == 0) {
            System.out.println("Processed " + logNumQuestions + " questions");
          }
          JSONObject qaJson = (JSONObject) qaObj;

          String idStr = (String) qaJson.get("id");
          String questionStr = (String) qaJson.get("question");
          TokenizedText questionTok = tokenize(questionStr, pipeline, knownWords, unknownWords, split, maxSentenceLength, jsonMetadata, verbose);
          assertReconstruction("question", questionStr, questionTok, -1, -1);
          putTokens(qaJson, questionTok, annotate);
          jsonMetadata.allTokens.addAll(questionTok.tokens);
          if (!hasAnswers) {
            numFullQuestions++;
            continue;
          }

          JSONArray answerObjs = (JSONArray) qaJson.get("answers");
          if (answerObjs.isEmpty()) {
            throw new RuntimeException("No answers\narticle:\n" + title + "\ncontext:\n" + contextStr + "\nid:\n" + idStr);
          }

          int numAnswers = answerObjs.size();
          int numInvalidAnswersAlign = 0;
          int numInvalidAnswersSentence = 0;
          for (Object answerObj : answerObjs) {
            JSONObject answerJson = (JSONObject) answerObj;
            String answerStr = (String) answerJson.get("text");
            int answerStartCharIdx = toIntExact((Long) answerJson.get("answer_start"));
            int answerAfterEndCharIdx = answerStartCharIdx + answerStr.length();
            int answerStartTokenIdx = contextTok.startCharIdxs.indexOf(answerStartCharIdx);
            int answerEndTokenIdx = contextTok.afterEndCharIdxs.indexOf(answerAfterEndCharIdx);
            if (verbose) {
              if (answerStartTokenIdx < 0) {
                print_bad_answer(true, idStr, contextTok, answerStr, answerStartCharIdx, answerAfterEndCharIdx);
              }
              if (answerEndTokenIdx < 0) {
                print_bad_answer(false, idStr, contextTok, answerStr, answerStartCharIdx, answerAfterEndCharIdx);
              }
            }

            if (answerStartTokenIdx < 0 || answerEndTokenIdx < 0) {
              answerJson.put("valid", false);
              numInvalidAnswersAlign++;
            } else {
              assertReconstruction("answer", answerStr, contextTok, answerStartTokenIdx, answerEndTokenIdx);
              answerJson.put("valid", true);
              answerJson.put("start_token_idx", answerStartTokenIdx);
              answerJson.put("end_token_idx", answerEndTokenIdx);
            }
          }
          jsonMetadata.numAnswers += numAnswers;
          jsonMetadata.numInvalidAnswersAlign += numInvalidAnswersAlign;
          jsonMetadata.numInvalidAnswersSentence += numInvalidAnswersSentence;

          if (numInvalidAnswersAlign + numInvalidAnswersSentence == 0) {
            numFullQuestions++;
          } else if (numInvalidAnswersAlign + numInvalidAnswersSentence < numAnswers) {
            numPartialQuestions++;
          } else {
            numInvalidQuestions++;
          }
        }
        jsonMetadata.numQuestions += numQuestions;
        jsonMetadata.numFullQuestions += numFullQuestions;
        jsonMetadata.numPartialQuestions += numPartialQuestions;
        jsonMetadata.numInvalidQuestions += numInvalidQuestions;
        if (numInvalidQuestions == numQuestions) {
          numInvalidParagraphs++;
        }
      }
      jsonMetadata.numParagraphs += numParagraphs;
      jsonMetadata.numInvalidParagraphs += numInvalidParagraphs;
      if (numInvalidParagraphs == numParagraphs) {
        numInvalidArticles++;
      }
    }
    jsonMetadata.numArticles += numArticles;
    jsonMetadata.numInvalidArticles += numInvalidArticles;

    JSONArray unknownWordsArray = new JSONArray();
    unknownWordsArray.addAll(unknownWords);
    json.put("unknown_words", unknownWordsArray);

    jsonMetadata.unknownTokens.addAll(unknownWords);
    return jsonMetadata;
  }


  private static TokenizedText tokenize(
    String s, StanfordCoreNLP pipeline, Set<String> knownWords, Set<String> unknownWords,
    boolean split, int maxSentenceLength, JsonMetadata jsonMetadata, boolean verbose) {

    Annotation annotation = new Annotation(s);
    pipeline.annotate(annotation);

    List<CoreLabel> coreLabels = annotation.get(CoreAnnotations.TokensAnnotation.class);

    List<List<CoreLabel>> rawSents = new LinkedList<List<CoreLabel>>();
    for (CoreMap sentence: annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
      List<CoreLabel> sentenceCoreLabels = sentence.get(CoreAnnotations.TokensAnnotation.class);
      rawSents.add(sentenceCoreLabels);
    }

    List<List<CoreLabel>> sents = fixSentences(rawSents, jsonMetadata, maxSentenceLength);
    int clIdx = 0;
    for (List<CoreLabel> sent : sents) {
      for (CoreLabel cl : sent) {
        if (cl != coreLabels.get(clIdx++)) { 
          throw new RuntimeException("Mis-aligned sentence splitting");
        }
      }
    }
    if (clIdx != coreLabels.size()) { 
      throw new RuntimeException("Invalid sentence splitting size");
    }

    TokenizedText tokenizedText = new TokenizedText();

    int sentIdx = -1;
    int numTokensPrevSents = 0;
    for (List<CoreLabel> sent : sents) {
      sentIdx++;
      for (CoreLabel coreLabel : sent) {
        String token = coreLabel.word();
        String original = coreLabel.originalText();
        String whitespaceAfter = coreLabel.after();
        int startCharIdx = coreLabel.beginPosition();
        int afterEndCharIdx = coreLabel.endPosition();
        String posTag = coreLabel.tag();
        String nerTag = coreLabel.ner();

        // attempt matching to known word
        String knownWord = tokenToKnownWord(token, knownWords);
        if (knownWord != null) {
          tokenizedText.tokens.add(knownWord);
          tokenizedText.originals.add(original);
          tokenizedText.whitespaceAfters.add(whitespaceAfter);
          tokenizedText.startCharIdxs.add(startCharIdx);
          tokenizedText.afterEndCharIdxs.add(afterEndCharIdx);
          tokenizedText.posTags.add(posTag);
          tokenizedText.nerTags.add(nerTag);
          tokenizedText.tokenSentenceIdxs.add(sentIdx);
          continue;
        }

        // attempt token splitting if configured to do so
        String[] subTokens = token.split("-");
        if (split && token.equals(original) && subTokens.length > 1 &&
          Arrays.stream(subTokens).noneMatch(subToken -> subToken.isEmpty())) {

          boolean useAlt = false;
          List<String> altTokens = new LinkedList<String>();
          List<String> altOriginals = new LinkedList<String>();
          List<String> altWhitespaceAfters = new LinkedList<String>();
          List<Integer> altStartCharIdxs = new LinkedList<Integer>();
          List<Integer> altAfterEndCharIdxs = new LinkedList<Integer>();
          List<String> altPosTags = new LinkedList<String>();
          List<String> altNerTags = new LinkedList<String>();
          List<Integer> altTokenSentenceIdxs = new LinkedList<Integer>();
          Set<String> unknownWordsToAdd = new HashSet<String>();
          int altCharIdx = startCharIdx;
          for (int i=0; i<subTokens.length; i++) {
            String altToken = tokenToKnownWord(subTokens[i], knownWords);
            if (altToken == null) {
              altToken = subTokens[i];
              unknownWordsToAdd.add(altToken);
            } else {
              useAlt = true;
            }
            altTokens.add(altToken);
            altOriginals.add(subTokens[i]);
            altWhitespaceAfters.add(i == subTokens.length-1 ? whitespaceAfter: "");
            altStartCharIdxs.add(altCharIdx);
            altCharIdx += subTokens[i].length();
            altAfterEndCharIdxs.add(altCharIdx);
            altPosTags.add(posTag);
            altNerTags.add(nerTag);
            altTokenSentenceIdxs.add(sentIdx);
            if (i < subTokens.length-1) {
              altTokens.add("-");
              altOriginals.add("-");
              altWhitespaceAfters.add("");
              altStartCharIdxs.add(altCharIdx);
              altCharIdx += 1;
              altAfterEndCharIdxs.add(altCharIdx);
              altPosTags.add("<split_null>");
              altNerTags.add("<split_null>");
              altTokenSentenceIdxs.add(sentIdx);
            }
          }
          if (altCharIdx != afterEndCharIdx) {
            throw new RuntimeException("Bad splitting");
          }
          if (useAlt) {
            tokenizedText.tokens.addAll(altTokens);
            tokenizedText.originals.addAll(altOriginals);
            tokenizedText.whitespaceAfters.addAll(altWhitespaceAfters);
            tokenizedText.startCharIdxs.addAll(altStartCharIdxs);
            tokenizedText.afterEndCharIdxs.addAll(altAfterEndCharIdxs);
            tokenizedText.posTags.addAll(altPosTags);
            tokenizedText.nerTags.addAll(altNerTags);
            tokenizedText.tokenSentenceIdxs.addAll(altTokenSentenceIdxs);
            unknownWords.addAll(unknownWordsToAdd);
            jsonMetadata.numSplits++;
            if (verbose) {
              System.out.println(String.join("", altTokens));
            }
            continue;
          }
        }

        // not found
        tokenizedText.tokens.add(token);
        tokenizedText.originals.add(original);
        tokenizedText.whitespaceAfters.add(whitespaceAfter);
        tokenizedText.startCharIdxs.add(startCharIdx);
        tokenizedText.afterEndCharIdxs.add(afterEndCharIdx);
        tokenizedText.posTags.add(posTag);
        tokenizedText.nerTags.add(nerTag);
        tokenizedText.tokenSentenceIdxs.add(sentIdx);
        unknownWords.add(token);
      } // end going over core labels
      tokenizedText.sentenceLengths.add(tokenizedText.tokens.size() - numTokensPrevSents);
      numTokensPrevSents = tokenizedText.tokens.size();
    } // end going over sents
    return tokenizedText;
  }


  private static String tokenToKnownWord(String token, Set<String> knownWords) {
    if (knownWords.contains(token)) {
      return token;
    }
    if (knownWords.contains(capitalize(token))) {
      return capitalize(token);
    }
    if (knownWords.contains(token.toLowerCase())) {
      return token.toLowerCase();
    }
    if (knownWords.contains(token.toUpperCase())) {
      return token.toUpperCase();
    }
    return null;
  }


  private static String capitalize(String s) {
    return Character.toUpperCase(s.charAt(0)) + s.substring(1);
  }

  private static void assertReconstruction(String name, String targetStr,
    TokenizedText tokenizedText, int reconstStartTokenIdx, int reconstEndTokenIdx) {
    if (reconstStartTokenIdx < 0 || reconstEndTokenIdx < 0) {
      reconstStartTokenIdx = 0;
      reconstEndTokenIdx = tokenizedText.tokens.size() - 1;
    }
    String reconstStr = tokenizedText.originals.get(reconstStartTokenIdx);
    for (int i=reconstStartTokenIdx+1; i<=reconstEndTokenIdx; i++) {
      reconstStr += tokenizedText.whitespaceAfters.get(i-1) + tokenizedText.originals.get(i);
    }
    String trimmedTargetStr = targetStr.trim();

    String msg = "";
    if (reconstStr.length() != trimmedTargetStr.length()) {
      msg = "Diff length: reconst " + reconstStr.length() + " target " + trimmedTargetStr.length() + "; ";
    } else {
      for (int i=0; i<reconstStr.length(); i++) {
        char rc = reconstStr.charAt(i);
        char tc = trimmedTargetStr.charAt(i);
        if (rc != tc && !(rc == 160 && tc == 32)) {
          // rc==160 and tc==32 : the Stanford tokenizer sometimes introduces 160 (non-breaking space) instead of 32 (plain space) e.g. in "8 1/2"
          msg += "charAt " + i + " reconst [" + (int)rc + "] target [" + (int)tc + "] '" + targetStr.substring(i-5, i+5) + "'; ";
        }
      }
    }
    if (!msg.isEmpty()) {
      throw new RuntimeException(
        "\nBad " + name + " reconstruction:\ntargetStr:\n[" + targetStr + "]\nreconstStr:\n[" + reconstStr + "]\n" + msg + "\n");
    }
  }


  private static List<List<CoreLabel>> fixSentences(List<List<CoreLabel>> rawSents, JsonMetadata jsonMetadata, int maxSentenceLength) {
    List<List<CoreLabel>> joinedSents = new LinkedList<List<CoreLabel>>();
    // Join sentences if needed
    List<CoreLabel> currSent = rawSents.get(0);
    joinedSents.add(currSent);
    for (int rawSentIdx=1; rawSentIdx < rawSents.size(); rawSentIdx++) {
      List<CoreLabel> nextSent = rawSents.get(rawSentIdx);
      if (joinSentences(currSent, nextSent)) {
        currSent.addAll(nextSent);
        jsonMetadata.numSentenceJoins++;
      } else {
        currSent = nextSent;
        joinedSents.add(currSent);
      }
    }
    // Artificially break very long sentences
    List<List<CoreLabel>> sents = new LinkedList<List<CoreLabel>>();
    for (List<CoreLabel> sent : joinedSents) {
      if (sent.size() > maxSentenceLength) {
        sents.addAll(breakSentence(sent, maxSentenceLength));
        jsonMetadata.numSentenceBreaks++;
      } else {
        sents.add(sent);
      }
    }
    return sents;
  }

  private static Set<String> fixedSuffixes = new HashSet<String>(Arrays.asList(new String[]{
    "Fr. ",
    "Rev. ",
    "St. ",
    "PT. ",
    "Rs. ",
    "Ss. ",
    "Ecl. ",
    "Bros. ",
    "Pharm. ",
    "Ps. ",
    "app. "
  }));
  private static Set<String> fixedPrefixes = new HashSet<String>(Arrays.asList(new String[]{
    ".gov",
    ".museum",
    ".com",
    ".net",
    ".NET",
    ".org",
    ".nf",
    ".mp3",
    ".ts",
    ".m2ts",
    ".ipg",
    ".zip",
  }));

  private static boolean joinSentences(List<CoreLabel> firstSent, List<CoreLabel> secondSent) {
    if (firstSent.size() < 2 || secondSent.size() < 1) {
      return false;
    }
    CoreLabel firstLastM1 = firstSent.get(firstSent.size() - 2);
    CoreLabel firstLast = firstSent.get(firstSent.size() - 1);
    CoreLabel secondFirst = secondSent.get(0);
    String firstSuffixStr = firstLastM1.originalText() + firstLastM1.after() + firstLast.originalText() + firstLast.after();
    String secondPrefixStr = firstLast.originalText() + firstLast.after() + secondFirst.originalText();
    return fixedSuffixes.contains(firstSuffixStr) || fixedPrefixes.contains(secondPrefixStr);
  }

  private static List<List<CoreLabel>> breakSentence(List<CoreLabel> sent, int maxSentenceLength) {
    List<List<CoreLabel>> sents = new LinkedList<List<CoreLabel>>();
    while (sent.size() > maxSentenceLength) {
      sents.add(sent.subList(0, maxSentenceLength));
      sent = sent.subList(maxSentenceLength, sent.size());
    }
    sents.add(sent);
    return sents;
  }

  @SuppressWarnings("unchecked")
  private static void putTokens(JSONObject jsonObj, TokenizedText tokenizedText, boolean annotate) {
    JSONArray tokens = new JSONArray();
    JSONArray originals = new JSONArray();
    JSONArray whitespaceAfters = new JSONArray();
    JSONArray sentenceLengths = new JSONArray();
    tokens.addAll(tokenizedText.tokens);
    originals.addAll(tokenizedText.originals);
    whitespaceAfters.addAll(tokenizedText.whitespaceAfters);
    sentenceLengths.addAll(tokenizedText.sentenceLengths);
    jsonObj.put("tokens", tokens);
    jsonObj.put("originals", originals);
    jsonObj.put("whitespace_afters", whitespaceAfters);
    jsonObj.put("sentence_lengths", sentenceLengths);
    if (annotate) {
      JSONArray posTags = new JSONArray();
      JSONArray nerTags = new JSONArray();
      posTags.addAll(tokenizedText.posTags);
      nerTags.addAll(tokenizedText.nerTags);
      jsonObj.put("pos_tags", posTags);
      jsonObj.put("ner_tags", nerTags);
    }
  }


  private static void print_bad_answer(boolean isStart, String idStr, TokenizedText contextTok,
    String answerStr, int answerStartCharIdx, int answerAfterEndCharIdx) {
    String msg = String.format(
      "\n%-4s%-20s%-20s%-9s%-9s",
      "idx", "word", "original", "beginPos", "endPos");
    List<Integer> contextCharIdxs = isStart ? contextTok.startCharIdxs : contextTok.afterEndCharIdxs;
    int answerCharIdx = isStart ? answerStartCharIdx : answerAfterEndCharIdx;

    int insertionIdx = -Collections.binarySearch(contextCharIdxs, answerCharIdx) - 1;
    int firstIdx = Math.max(0, insertionIdx - 3);
    int lastIdx = Math.min(contextCharIdxs.size() - 1, insertionIdx + 3);
    for (int i=firstIdx; i<=lastIdx; i++) {
      msg += String.format(
        "\n%-4d%-20s%-20s%-9d%-9d",
        i, "["+contextTok.tokens.get(i)+"]", "["+contextTok.originals.get(i)+"]",
        contextTok.startCharIdxs.get(i), contextTok.afterEndCharIdxs.get(i));
    }
    String msgTitle = String.format(
      "\nidStr: %s\nisStart: %b\nanswerStr: %s\nanswerStartCharIdx: %d\nanswerAfterEndCharIdx: %d\ninsertionIdx: %d\n",
      idStr, isStart, answerStr, answerStartCharIdx, answerAfterEndCharIdx, insertionIdx);
    msg += msgTitle;
    System.out.println(msg);
  }
}

