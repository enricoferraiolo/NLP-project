import tensorflow as tf

# Convert a tokenized dataset to a TensorFlow DataLoader
def create_tf_dataloader(tokenized_dataset, batch_size=32):
    def gen():
        for example in tokenized_dataset:
            yield (
                {
                    "input_ids": example["input_ids"],
                    "attention_mask": example["attention_mask"],
                },
                example["labels"],
            )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
