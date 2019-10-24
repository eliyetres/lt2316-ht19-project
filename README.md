# Project for Machine Learning course at the University of Gothenburg

Abstractive text summarization using Sequence-to-Sequence modeling with attention mechanism.

## Dataset

The CNN News Story Dataset

The DeepMind Q&A Dataset is a large collection of news articles from CNN and the Daily Mail with associated summaries. The dataset was developed as a question and answering task for deep learning and was presented in the 2015 paper "Teaching Machines to Read and Comprehend." The dataset is commonly used in text summarization tasks.

### Preprocessing

The dataset was downloaded and extraxted from DeepMind Q&A Dataset. The CNN dataset has in total 92580 news article stories. The dataset has full story text followed by a number of "highlight" points that will be used as the summary. Pre-processing was done by lowercasing all words and removing punctuatation.

## Approach

This is an abstraction-based approach to text summarizatrion that uses a Seq2Seq encoder-decoder network to generate new shorter sentences from the original ones. The CNN Mail dataset has news articles and summarizations that is commonly used to evaluate summarization, and the results can be evaluated using ROGUE-metrics.

### Model implementation

* Encoder: Bi-directional LSTM layer that extracts information from the original text. The LSTM reads one word at a time, updates its hidden state based on the current word and the words it has read before. The encoder converts the sentence into a vector of features.

* Attention: is placed on the encoder features to make them even more specific to the current word. Attention model — Without attention, the input to decoder is the final hidden state from the encoder which is too small to contain all the neccesary information, so it becomes an information bottleneck. Through attention mechanism, the decoder can access the intermediate hidden states in the encoder and use all that information to decide which word is next.

* Decoder: Uni-directional LSTM layer that generates summaries one word at a time. The decoder LSTM starts working once it gets the signal than the full source text has been read. It uses information from the encoder as well as what is has written before to create the probability distribution over the next word. It takes the vector of features from the encoder and predicts each word based on the previous word prediction and the output.

Encoder and Decoder are the main parts but thisarchitecture in itself without attention isn't very successful. They have two shortcomings: they are liable to reproduce factual details inaccurately, and they tend to repeat themselves.

## Training

## Testing

## Evaluation

### ROUGE-metrics

(Recall-Oriented Understudy for Gisting Evaluation) metric.  
ROUGE-n is recall between candidate summary n-grams and n-grams from reference summary.

* ROUGE-1

* ROUGE-2

* ROUGE-L
