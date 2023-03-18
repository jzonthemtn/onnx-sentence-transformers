# Sentence Transformers ONNX

This was built for [OPENNLP-1442](https://issues.apache.org/jira/projects/OPENNLP/issues/OPENNLP-1442) and the 2023 Linux Foundation Open Source Summit.

Models from https://huggingface.co/optimum/all-MiniLM-L6-v2/tree/main

## Input

The inputs match except for the token_type_ids. ONNX Runtime complains in Java that it's (`1`) a non-zero value.

### Java

```
101 2577 2899 2001 2343 102
```

### Python

```
{'input_ids': tensor([[ 101, 2577, 2899, 2001, 2343,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}

```

## Output

### Java

```

```

### Python

Python output is consistent whether its using Sentence Transformers or Hugging face (the two methods at https://huggingface.co/optimum/all-MiniLM-L6-v2).

The output is consistent when padding and truncation are both `False`. The difference in the Java/Python outputs appears to be the `mean_pooling` and normalization in the Python code.

```
[[ 2.09840528e-05  8.30730423e-02  8.77965987e-02  2.76824255e-02
...
   2.49170326e-02  6.89062029e-02  3.37986946e-02 -9.98165272e-03]]
```