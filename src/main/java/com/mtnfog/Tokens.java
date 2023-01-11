package com.mtnfog;

public class Tokens {

  private String[] tokens;
  private long[] ids;
  private long[] mask;
  private long[] types;

  /**
   * Creates a new instance to hold the tokens for input to an ONNX model.
   * @param tokens The tokens themselves.
   * @param ids The token IDs as retrieved from the vocabulary.
   * @param mask The token mask. (Typically all 1.)
   * @param types The token types. (Typically all 1.)
   */
  public Tokens(String[] tokens, long[] ids, long[] mask, long[] types) {

    this.tokens = tokens;
    this.ids = ids;
    this.mask = mask;
    this.types = types;

  }

  public String[] getTokens() {
    return tokens;
  }

  public long[] getIds() {
    return ids;
  }

  public long[] getMask() {
    return mask;
  }

  public long[] getTypes() {
    return types;
  }

}