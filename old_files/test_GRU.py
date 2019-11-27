import torch
import matplotlib.pyplot as plt
import config
import math
from torchtext.data import BucketIterator, Field, TabularDataset
from utils import tokenize_en, evaluate
import torch.nn as nn
import matplotlib.ticker as ticker
import random

model = torch.load('model-GRU.pt')
SRC = torch.load("SRC.pt")
TRG = torch.load("TRG.pt")

pad_idx = SRC.vocab.stoi['<pad>']
#print(pad_idx)
fields = [('stories', SRC), ('highlights', TRG)]

### train
train_data = TabularDataset(
	path=config.TRAIN_DATA_PATH, format='csv', fields=fields, skip_header=True)
#### Test

test_data = TabularDataset(
	path=config.TEST_DATA_PATH, format='csv', fields=fields, skip_header=True)
test_iterator = BucketIterator(test_data, sort_key=lambda x:(len(x.stories), len(x.highlights)), batch_size=1, device=config.DEVICE, sort_within_batch=True, train=False, shuffle=False)

criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=pad_idx)





#test_loss = evaluate(model, test_iterator, criterion)

#print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


def translate_sentence(model, src, tokenized_sentence):
    model.eval()
    tokenized_sentence = [config.SOS_TOKEN] + [t.lower() for t in tokenized_sentence] + [config.EOS_TOKEN]
    tokenized_sentence = [value for value in tokenized_sentence if value not in ["[","]"]]
    numericalized = [SRC.vocab.stoi[t] for t in tokenized_sentence] 
    sentence_length = torch.LongTensor([len(numericalized)]).to(config.DEVICE) 

    
    tensor1 = torch.LongTensor(numericalized).unsqueeze(1).to(config.DEVICE) 


    print(tensor1.size(), tensor1.type(), type(tensor1))
    


    print(tokenized_sentence)


    # output = model(src, trg)
    translation_tensor_logits, attention = model(tensor1, sentence_length) # error here
    
    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)


    translation = [TRG.vocab.itos[t] for t in translation_tensor]
    translation, attention = translation[1:], attention[1:]

    print("translation")
    print(translation)
    return translation, attention


# def display_attention(sentence, translation, attention):
    
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111)
    
#     attention = attention.squeeze(1).cpu().detach().numpy()    
#     cax = ax.matshow(attention, cmap='bone')   
#     ax.tick_params(labelsize=15)
#     ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
#                        rotation=45)
#     ax.set_yticklabels(['']+translation)

#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     plt.show()
#     plt.close()

example_idx = 100

src = vars(train_data.examples[example_idx])['stories']
trg = vars(train_data.examples[example_idx])['highlights']

#print(f'src = {src}')
#print(f'trg = {trg}')

#translation, attention = translate_sentence(model, src)

#print(f'predicted trg = {translation}')




#display_attention(src, translation, attention)

example_idx = 35

src = vars(test_data.examples[example_idx])['stories']
trg = vars(test_data.examples[example_idx])['highlights']

#print(f'src = {src}')
#print(f'trg = {trg}')



for i, batch in enumerate(test_iterator):        
        model.eval()
        src = batch.stories
        trg = batch.highlights
        tokenized_story = vars(test_data.examples[i])['stories']
        tokenized_highlight = vars(test_data.examples[i])['highlights']

        sentence_length = torch.LongTensor([len(src)]).to(config.DEVICE) 

        translation_tensor_logits, attention = model(src, None, 0) # error here



    
        translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)

        print(translation_tensor)
        translation = [TRG.vocab.itos[t] for t in translation_tensor]
        print(translation)
        translation, attention = translation[1:], attention[1:]

        print(translation)


        #translation, attention = translate_sentence(model, src,tokenized_sentence)

       