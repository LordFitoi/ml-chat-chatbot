from core.corpus import Corpus
from core.encoder import StringEncoder
from core.generator import TextGeneratorModel

corpus = Corpus("corpus/")
string_encoder = StringEncoder()
x_train, y_train = string_encoder.train(corpus)

model = TextGeneratorModel(string_encoder)
model.fit(x_train, y_train)

while True:
    context = input(">> ")
    print("BOT:", model(context))
