
import spacy
import numpy as np
from spacy.language import Language
from tensorflow.keras import layers

class StringEncoder:
    def __init__(self):
        self.tokenizer = spacy.load("es_core_news_md")
        self.tokenizer.add_pipe("set_break_lines", before="parser")
        self.embedding = layers.StringLookup(output_mode="one_hot")

    @Language.component("set_break_lines")
    def set_break_lines(doc):
        for token in doc[:-1]:
            doc[token.i + 1].is_sent_start = "\n" in token.text
        return doc

    def set_pos_encode(self, x, i):
        positional = np.full(len(x), i)
        frequency = np.arange(1, len(x) + 1)
        return x + np.sin(positional / frequency)

    def decode(self, token_vector):
        token_index = np.argmax(token_vector)
        return self.embedding.get_vocabulary()[token_index]

    def train(self, corpus):
        x1_train = []
        x2_train = []
        y_train = []

        vocabulary = ["[START]", "[END]"]

        start_token = self.tokenizer("[START]")
        end_token = self.tokenizer("[END]")

        for document in corpus.documents:
            sentences = [sent for sent in self.tokenizer(document).sents]

            for i in range(len(sentences) - 1):
                context = sentences[i]
                response = [start_token] + [
                    token for token in sentences[i + 1]
                    if "\n" not in token.text] + [end_token]
                                
                if (not context):
                    continue

                for j in range(len(response) - 1):
                    token = response[j + 1].text

                    x1_train.append(context.vector)
                    x2_train.append(self.set_pos_encode(response[j].vector, j + 1))
                    y_train.append(token)
        
                    if token not in vocabulary and token not in ("[START]", "[END]"):
                        vocabulary.append(token)
        
        self.embedding.set_vocabulary(vocabulary)

        x_train = [np.array(x1_train), np.array(x2_train)]
        y_train = np.array([self.embedding(token) for token in y_train])

        return x_train, y_train
    
    def __call__(self, sentence):
        return self.tokenizer(sentence).vector
