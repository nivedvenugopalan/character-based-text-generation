# imports
import os
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import json
import numpy as np
from train import train
from generate import generate

print("Starting character based generation...")


def generate_config(datapath, modelname):
    # return a dict with all the configs parameters
    return {
        "epochs": 100,
        "seq_length": 100,
        "batch_size": 64,
        "rnn_units": 1024,
        "datapath": datapath,
    }


def get_model_names(configs):
    return list(configs["models"].keys())


def save(configs):
    # save the configs file
    with open('./src/data/configs.json', 'w') as f:
        json.dump(configs, f)


while True:
    # open the configs file
    with open('./src/data/configs.json') as f:
        configs = json.load(f)

    print("""1. Train a new model
2. Generate text
3. Exit""")

    choice = input("Enter your choice: ")

    if choice == "1":

        # get the modelname
        while True:
            modelname = input(
                "Enter the name of the new-model (Enter 0 to exit): ")

            if modelname == "0":
                break

            # check if model already exists
            if modelname in list(configs.keys()):
                print("Model already exists!")
                print("Please choose another name")
                print()

            else:
                break

        # get the datapath of the data file
        datapath = input("Enter the path to the data file (.txt file): ")

        # get the configs
        generated_config = generate_config(datapath, modelname)
        configs["models"][modelname] = generated_config

        # print the model and ask the user if it is ok
        while True:
            print("1. Model name:", modelname)
            print("2. Data path:", datapath)
            print("3. Epochs:", configs["models"][modelname]["epochs"])
            print("4. Sequence length:",
                  configs["models"][modelname]["seq_length"])
            print("5. Batch size:", configs["models"][modelname]["batch_size"])
            print("6. RNN units:", configs["models"][modelname]["rnn_units"])
            print()

            ok = input("Is this ok? (y/n): ")

            if ok == "y":
                break

            else:
                change = int(
                    input("What would you like to change? (2-6, n): "))

                if change == 2:
                    datapath = input(
                        "Enter the path to the data file (.txt file): ")

                elif change == 3:
                    configs["models"][modelname]["epochs"] = int(
                        input("Enter the number of epochs: "))

                elif change == 4:
                    configs["models"][modelname]["seq_length"] = int(
                        input("Enter the sequence length: "))

                elif change == 5:
                    configs["models"][modelname]["batch_size"] = int(
                        input("Enter the batch size: "))

                elif change == 6:
                    configs["models"][modelname]["rnn_units"] = int(
                        input("Enter the number of RNN units: "))

                elif change == "n":
                    pass

                print()

        # train the model
        print("Training the model...")
        _, vocab, char2idx, idx2char = train(datapath, modelname, configs["models"][modelname]["epochs"], configs["models"]
                                             [modelname]["seq_length"], configs["models"][modelname]["batch_size"], configs["models"][modelname]["rnn_units"])
        print("Model trained!")
        del _

        # save the vocab, char2idx and idx2char
        configs["models"][modelname]["vocab"] = vocab
        configs["models"][modelname]["char2idx"] = char2idx
        configs["models"][modelname]["idx2char"] = idx2char.tolist()

        # save the configs
        save(configs)

        print()

    elif choice == "2":
        # choose the model

        # print all the models
        print("Available models:")
        for i in range(len(get_model_names(configs))):
            print(f"{i + 1}. {get_model_names(configs)[i]}")

        # get the model
        model_id = int(input("Enter the number of the model: "))

        modelname = get_model_names(configs)[model_id - 1]

        # generate text
        starting_text = input("Enter the starting text: ")

        # generate text
        generated = generate(
            len(configs["models"][modelname]["vocab"]),
            configs["models"][modelname]["rnn_units"],
            modelname,
            starting_text,
            configs["temperature"],
            configs["models"][modelname]["char2idx"],
            np.array(configs["models"][modelname]["idx2char"]),
        )
        print(generated)

    elif choice == "3":
        # save the configs file
        save(configs)

        break
