import numpy as np
from tensorflow.keras import layers, optimizers, metrics, Model

class TextGeneratorModel:
    def __init__(self, string_encoder):
        self.string_encoder = string_encoder
        self.vector_length = 300

        vocab_shape = len(self.string_encoder.embedding.get_vocabulary())
        self.model = self.create_model(self.vector_length, vocab_shape)

    def create_model(self, input_shape, output_shape, hidden_shape=500):
        # ENCODER
        enc_inputs = layers.Input(shape=input_shape)
        enc_hidden_1 = layers.Dense(hidden_shape, activation="relu")(enc_inputs)
        enc_hidden_2 = layers.Dense(hidden_shape, activation="relu")(enc_hidden_1)
        enc_outputs = layers.Dense(input_shape, activation="softplus")(enc_hidden_2)

        # DECODER
        dec_inputs = layers.Input(shape=input_shape)
        dec_add = layers.Add()([enc_outputs, dec_inputs])
        dec_hidden_1 = layers.Dense(hidden_shape, activation="relu")(dec_add)
        dec_hidden_2 = layers.Dense(hidden_shape, activation="relu")(dec_hidden_1)      
        dec_outputs = layers.Dense(output_shape, activation="softmax")(dec_hidden_2)
        
        # MODEL
        model = Model(inputs=(enc_inputs, dec_inputs), outputs=dec_outputs)
        model.compile(
            optimizer=optimizers.Adam(),
            loss="binary_crossentropy",
            metrics=[
                metrics.Recall()
            ]
        )
        return model

    def fit(self, x_train, y_train, epochs=100):
        self.model.fit(x_train, y_train, epochs=epochs)

    def token_encode(self, token, i):
        return self.string_encoder.set_pos_encode(
            self.string_encoder(token), i)

    def token_decode(self, x_input1, x_input2):
        x_input = [np.array([x_input1]), np.array([x_input2])]
        return str(self.string_encoder.decode(self.model(x_input)[0]))

    def __call__(self, context, max_length=20, noise=1):
        random_vector = np.random.uniform(-noise, noise, self.vector_length)
        context_vector = self.string_encoder(context) + random_vector

        token_vector = self.token_encode("[START]", 1)
        output = []

        for i in range(1, max_length):
            token = self.token_decode(context_vector, token_vector)
            if token == "[END]": break

            token_vector = self.token_encode(token, i + 1)
            output.append(token)
        
        return  " ".join(output)