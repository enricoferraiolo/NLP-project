# Encoder - Decoder model with LSTM

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(hidden_dim, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, h, c = self.lstm(x)
        return output, h, c

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size, activation='softmax')

    def call(self, x, enc_states):
        x = self.embedding(x)
        output, h, c = self.lstm(x, initial_state=enc_states)
        x = self.fc(output)
        return x, h, c

class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, encoder_input, decoder_input):
        """
        Passa l'input dell'encoder e del decoder attraverso il modello.
        """
        _, enc_hidden_h, enc_hidden_c = self.encoder(encoder_input)
        dec_output, _, _ = self.decoder(decoder_input, [enc_hidden_h, enc_hidden_c])
        return dec_output

import tensorflow as tf

def train_model(dataset, vocab_size, embedding_dim, hidden_dim, epochs, learning_rate):
    # Inizializza Encoder, Decoder e modello Seq2Seq
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder)


    # Compila il modello
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Loop di addestramento
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch, (inputs, targets) in enumerate(dataset):
            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]
            
            # print("INPUTS:")
            # print(inputs)
            with tf.GradientTape() as tape:
                predictions = model(inputs["input_ids"], decoder_input)
                loss = loss_fn(decoder_target, predictions)
            
            # Aggiorna i pesi
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if batch % 10 == 0:
                print(f"Batch {batch}: Loss = {loss.numpy()}")
