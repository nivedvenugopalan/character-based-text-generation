from train import build_model
import tensorflow as tf


def generate(vocabsize, rnnunits, modelname, starting_string, temperature, char2idx, idx2char):
    # build the model
    model = build_model(vocabsize, 256, rnnunits, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(
        f"./src/data/models/{modelname}/"
    ))

    model.build(tf.TensorShape([1, None]))

    def generate_text(model, start_string):
        # Evaluation step (generating text using the learned model)

        # Number of characters to generate
        num_generate = 800

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.

        # Here batch size == 1
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension

            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(
                predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))

    return generate_text(model, starting_string)
