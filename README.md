# Word-based Astractive Text Summarization using Sequence-to-Sequence modeling with attention mechanism

Project for Machine Learning course at the University of Gothenburg

## Dataset

The CNN News Story Dataset

The DeepMind Q&A Dataset is a large collection of news articles from CNN and the Daily Mail with associated summaries. The dataset was developed as a question and answering task for deep learning and was presented in the 2015 paper "Teaching Machines to Read and Comprehend." The dataset is commonly used in text summarization tasks.

### Preprocessing

The dataset was downloaded and extraxted from DeepMind Q&A Dataset. The CNN dataset has in total 92580 news article stories. The dataset has full story text followed by a number of "highlight" points that will be used as the summary.

All ``.story`` documents are read from a selected directory using `load_stories` and `load_doc`. `split_story` splits a document into news story and highlights and strips all empty lines. `clean_lines` starts by finding the beginning of the CNN article by matching the string "CNN", the string is later discarded and for the rest of the text, dashes and slashes are replaced by spaces to avoid compund words. The text is then lowercased and contractions are replaced with full words. Stopwords were removed from stories but kept for highlights. Finally, all other non-characters or numbers are removed using regex. The stories are tokenized and cut at a selected length (default lengths are 400 for stories and 100 for summaries).

### Splitting and saving data

The data is split into train, test and validation sets using sklearn's ``train_test_split``. The training data is 80% of the total data, and the validation data is 20% of the training data. Train, test, and validation sets are saved as pandas objects and before wrriting the data to csv files, if there is an epty line in either the stories or highlights, both lines are dropped.

## Approach

This is an abstraction-based approach to text summarizatrion that uses a Seq2Seq encoder-decoder network to generate new shorter sentences from the original ones. The CNN Mail dataset has news articles and summarizations that is commonly used to evaluate summarization, and the results can be evaluated using ROGUE-metrics.

## Model implementation

### Seq2Seq Model

The Seq2Seq model takes a variable-length sequence as an input, and returns a variable-length sequence as an output using a fixed-sized model. This is achieved by using two separate Recurrent Neural Networks (RNNs) added together. One RNN acts as an encoder, which encodes a variable length input sequence to a fixed-length context vector. This context vector (the final hidden layer of the RNN)  contains semantic information about the sentence that is input to the model. The second RNN is a decoder, which takes an input word and the context vector, and returns a prediction for the next word in the sequence and a hidden state to use in the next iteration.

### Encoder

The encoder RNN iterates through the input sentence one token (here, a word) at a time, at each time step outputting an output vector and a hidden state vector. The hidden state vector is then passed to the next time step, while the output vector is recorded. The encoder uses a bidirectional GRU layer, which means there are essentially two independent RNNs, one that is fed the input sequence in sequential order, and one in that's fed in reverse order. Using a bidirectional GRU makes it possible to encoding both past and future context.

An embedding layer is used to encode the word indices. The layer maps each word to a feature space of the hidden_size.
The words of the sentences are converted into indices and fed into the embedding layer. When trained, these values will encode semantic similarity between similar meaning words. The batch sequences are then padded and fed into the RNN module, using `nn.utils.rnn.pack_padded_sequence` and `nn.utils.rnn.pad_packed_sequence`. The data is then fed trough the forward pass to the bidirectional GRU layer, the outputs are summed and returned togerather with the final hidden state.

### Attention

The output attention module `Attn`'s weights have the same shape as the input sequence, which allows for multiplication by the encoder outputs. It returns a weighted sum which indicates which parts of encoder output to pay attention to. The attention calculated using the decoder’s current hidden state and the encoder’s outputs. The output attention weights have the same shape as the input sequence, allowing us to multiply them by the encoder outputs, returning a weighted sum which indicates the parts of encoder output to pay attention to.  The attention layer provides various methods to calculate the attention energies between the encoder output and decoder output which are names as "score functions" in the model's parameters. The output of the module is a softmax normalized weights tensor of shape ``[batch_size, 1, max_length]``.

### Decoder

The decoder RNN generates the response sentence token-by-token. It uses the encoder's vectors and internal hidden states to generate the next word in the sequence. It continues generating words until it outputs an `EOS_token`, (End of String) representing the end of the sentence. One batch is fed into the decoder one time step at a time. This means that the embedded word tensor and GRU output both have the shape ``[1, batch_size, hidden_size]``.

The word embedding for the input word is fed forward through unidirectional GRU. The attenrion weight ss from the current GRU output are calculated and multiplied with  the encoder outputs to get the weighted sum context vector. The weighted context vector and GRU output are concatenated, and using the attention layer the next word is predicted. It retuns the output and final hidden state.

## Training

The data is fed into the model using batches. To accommodate sentences of different sizes in the same batch, the batched input tensor is of shape ``[max_length, batch_size]``, where sentences shorter than the max_length are padded with a `PAD_token`. Converting  the sentences to tensors by mapping words to their indices using `indexesFromSentence` and pad, would make the tensor of shape ``[batch_size, max_length]`` and indexing the first dimension would return a full sequence across all time-steps. To be able to index our batch along time, and across all sequences in the batch, the input batch shape is transposed to (max_length, batch_size), This is done in the function  `zeroPadding`.

The `inputVar` function handles the process of converting sentences to tensors by creating a correctly shaped tensor padded with zeroes to match the max length. It also returns a tensor of lengths for each of the sequences in the batch which will be passed to the decoder. When all samples have the same length, the model must be informed that some part of the data is actually padding and should be ignored. This is done by using masking. The `outputVar` function performs a similar function to `inputVar`, but instead of returning a lengths tensor, it returns a binary mask tensor and a maximum target sentence length. The binary mask tensor has the same shape as the output target tensor, but every element that is a `PAD_token` is 0 and all others are 1. The mask tells the model that all 0 tokens should be ignored. `batch2TrainData` then takes a pairs of stories and highlights and returns the input and target tensors using the mentioned functions.

When dealing with batches of padded sequences, the `PAD_token` needs to be masked when calculating the loss. `maskNLLLoss`calculates the loss based on our decoder’s output tensor, the target tensor, and a binary mask tensor describing the padding of the target tensor. This loss function calculates the average negative log likelihood of the elements that correspond to a 1 in the mask tensor (representing the unmasked tokens).

Teacher forcing is used in the model and set to 0.5. The model uses gradient clipping to avoid the exploding gradient problem.

## Testing

### Decoding

Greedy decoding is the decoding method that is used during training when the model is *not* using teacher forcing. In other words, for each time step, the word eith the highest softmax value is chosen from the decoder output. This decoding method is optimal on a single time-step level. The class `GreedySearchDecoder` takes an input sequence ``[input_seq]`` of shape ``[input_seq length, 1]``, a scalar input length ``[input_length]`` tensor, and a max_length to bound the response sentence length. First, the sentence is encoded and padded, and fed trough the encoder model. The encoder's final hidden layer is the first hidden input to the decoder. The decoder's first input is the `SOS_token`, followed by the tensors to append decoded words to. One word is decoded at a time and fed forward through the decoder, returning the most likely word token and its softmax score. The model records the token and the score, and then prepares the current token to be the next input to the decoder. In the end it returns a collections of word tokens and scores.

## Evaluation

The evaluation function takes a sentence with batch_size 1. The words of the sentence is mapped to their corresponding indices, and transposed the dimensions to prepare the tensor for the models. When testing, the lengths are scalar values because the model only evaluates one sentence at a time. After this, decoded response sentence  tensor is fetched using the `GreedySearchDecoder` searcher. Finally the response’s indices are converted to words and returns the list of decoded words. `evaluateInput` A sentence from the test data is fed to the evaluate function to obtain a decoded output sentence. Tokens not in the vocabulary will be mapped to the `UNK_token`.

### ROUGE-metrics

(Recall-Oriented Understudy for Gisting Evaluation) metric.  
ROUGE-n is recall between candidate summary n-grams and n-grams from reference summary.

* ROUGE-1

* ROUGE-2

* ROUGE-L
