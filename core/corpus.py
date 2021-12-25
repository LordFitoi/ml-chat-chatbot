import re
from os import walk, path

class Corpus:
    def __init__(self, corpus_path):
        self.documents = []

        for filename in next(walk(corpus_path), (None, None, []))[2]:
            file_path = path.join(corpus_path, filename)
            self.load(file_path)
    
    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as txtfile:
            self.documents.append(txtfile.read())