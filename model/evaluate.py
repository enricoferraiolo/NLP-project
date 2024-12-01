def predict(encoder, decoder, input_seq, tokenizer, max_len):
    enc_output, h, c = encoder(input_seq)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    for t in range(max_len):
        predictions, h, c = decoder(dec_input, [h, c])
        predicted_id = tf.argmax(predictions[0, -1]).numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            break
        dec_input = tf.expand_dims([predicted_id], 0)
    return " ".join(result)
