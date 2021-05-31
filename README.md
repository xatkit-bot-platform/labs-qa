# labs-qa

## TO DO:

* beginPosition and endPosition of the answers are expressed in terms of tokens (e.g. beginPosition = 10 means that the answers starts at the 10th token of the given corpus). Try to convert it to char positions (each language model may have a different tokenizer and therefore the tokens for the same corpus may be different for 2 language models)
* Get the score of the answers (probabilities). Actually the score is composed by 2 scores: the score of the beginning of the answer and the score of the end of the answer.
* To solve: sometimes the best scored beginning is after the best scored end (e.g. best possible beginning of the answer at position 10 and best possible end of the answer at position 5), so the result is an empty answer. Try to get the second (or third, forth...) best option that contains an answer
* Deal with large corpus. Some transformers have a limit of 512 tokens, others 4096... When the given corpus exceeds this limit, we need to split it. There are different ways to do this, so we need to analyze the possibilities.
