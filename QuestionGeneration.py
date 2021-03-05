
import pickle
import spacy
import os
from fuzzywuzzy import fuzz
from random import randint
import tensorflow as tf

from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

epochs = 50  #@param {type: "number"} {type: "slider", min: 1, max: 100}
batch_size = 90 #@param {type:"slider", min:10, max:500, step:10}
rnn_size = 512 #@param {type: "number"}
num_layers = 5 #@param {type: "number"}
learning_rate = 0.005 #@param {type: "number"}
keep_probability = 0.8 #@param {type: "number"}
beam_width = 20 #@param {type: "number"}


# Load Spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en import STOP_WORDS
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

def load_pickle(filename):
    """Loads up the pickled dataset for further parsing and preprocessing"""
    documents_f = open('./files/'+filename+'.pickle', 'rb')
    data = pickle.load(documents_f)
    documents_f.close()
    
    return data
        
def clean_text(text, replace_entities=True):
    """Cleans the text in the same way as in data preprocessing part before training"""
    if replace_entities:
        spacy_text = nlp(text)
        text_ents = [(str(ent), str(ent.label_)) for ent in spacy_text.ents]
        
        text = text.lower()
        # Replace entities
        for ent in text_ents:
            replacee = str(ent[0].lower())
            replacer = str(ent[1])
            try:
                text = text.replace(replacee, replacer)
            except:
                pass
    else:
        text = text.lower()
        
    spacy_text = nlp(text)
    spacy_text = [str(token.orth_) for token in spacy_text 
                  if not token.is_punct and not token.is_stop]
    spacy_text = ' '.join(spacy_text)

    return spacy_text

def accuracy(a, b):
  return fuzz.ratio(a.lower(), b.lower())

def text_to_seq(input_sequence):
    """Prepare the text for the model"""
    text = clean_text(input_sequence)
    return [vocab2int.get(word, vocab2int['<UNK>']) for word in text.split()]

int2vocab = load_pickle('int2vocab')
vocab2int = load_pickle('vocab2int')
dev_squad_paragraphs = load_pickle('dev_squad_paragraphs')
print("load")

dev_squad_paragraphs = list(set(dev_squad_paragraphs))
# random_example = randint(0, len(dev_squad_paragraphs))
random_example = randint(0, len(dev_squad_paragraphs)-1)
input_sequence = dev_squad_paragraphs[5]
print(dev_squad_paragraphs[5])



text = text_to_seq(input_sequence)
checkpoint_path = './files/model.ckpt'

loaded_graph = tf.Graph()





with tf.Session(graph=loaded_graph) as sess:
    
    
    # Load saved model
    try:
        print('Restoring old model from %s...' % checkpoint_path)
    
        loader=tf.train.import_meta_graph(checkpoint_path + ".meta")
        print('Restoring')
        loader.restore(sess, checkpoint_path)
        print('Restored')
    except Exception as e: 
        print(e)
        
 
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    input_length = loaded_graph.get_tensor_by_name('input_length:0')
    target_length = loaded_graph.get_tensor_by_name('target_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                      target_length: [25], 
                                      input_length: [len(text)]*batch_size,
                                      keep_prob: 1.0})

# Remove the padding from the tweet
pad = vocab2int["<PAD>"] 
new_logits = []
for i in range(batch_size):
    new_logits.append(answer_logits[i].T)

print('Original Text:', input_sequence.encode('utf-8').strip())

generated_question = ""
print('\nGenerated Questions:')
for index in range(beam_width):
    print(' -- : {}'.format(" ".join([int2vocab[i] for i in new_logits[1][index] if i != pad and i != -1])))
    

print(accuracy("Supplies from Jacksonville were in support of which faction in the Civil War?  What was the name of the battle that marked the first Confederate win in Florida? After what battle did Union forces return to and occupy Jacksonville for the rest of the war? What factors negatively impacted Jacksonville following the war? In what year was the battle that resulted from a Confederate cavalry unit attacking a Union expedition? Who did Jacksonville support with supplies during the Revolutionary war? What was the name of the battle that marked the first Confederate loss in Florida? After what battle did Union forces leave Jacksonville for good? What battle involve the Confederate Calvary in 1862? During what word was Jacksonville a key supply point for the North?","Supplies from Jacksonville were in support of which faction in the Civil War?  What was the name of the battle that marked the first Confederate win in Florida? After what battle did Union forces return to and occupy Jacksonville for the rest of the war? What factors negatively impacted Jacksonville following the war? In what year was the battle that resulted from a Confederate cavalry unit attacking a Union expedition? Who did Jacksonville support with supplies during the Revolutionary war? What was the name of the battle that marked the first Confederate loss in Florida? After what battle did Union forces leave Jacksonville for good? What battle involve the Confederate Calvary in 1862? During what word was Jacksonville a key supply point for the North?",generated_question))

"""Form Feilds input

"""
