# 1. Import Libraries and initialize variables
import keras
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pickle

# initialize all variables
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# Parsing the dataset file
# Teacher Forcing Algorithm (TFA) network model uses ground truth input rather than output from the previous model

# 2. reading the dataset
with open('eng-french.txt', 'r', encoding="utf-8") as f:
    rows = f.read().split('\n')

# read first 10,000 rows
for row in rows[:10000]:
    # split the input text and the target, has '/t' for seperation
    input_text, target_text = row.split('\t')

    # add '\t' at the start and '\n' at the end of text
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text.lower())  # append the input text to the list created
    target_texts.append(target_text.lower())

    # Split character from text & add in respective sets
    input_characters.update(list(input_text.lower()))  # adding the list of input texts to the set
    target_characters.update(list(target_text.lower()))

# sort the input and target characters
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

# len of input and target characters
num_en_chars = len(input_characters)
num_dec_chars = len(target_characters)

# getting max length of target and input texts i.e. max element of the list
max_input_length = max([len(i) for i in input_texts])
max_target_length = max([len(i) for i in target_texts])

print("Number of encoder characters: ", num_en_chars)
print("Number of decoder characters: ", num_dec_chars)

print("Maximum input length: ", max_input_length)
print("Maximum target length: ", max_target_length)


# We will get total number of input(encoder) and output(decoder) characters, max len of input texts and of target texts


# 3. One Hot Encoding (Vectorizer): Encode data in binary format

# 3D array of
# enc input data => Number of Pairs, Max Length of English text, Number of English text characters
# dec input data => Number of Pairs, Max Length of French text, Number of French text characters
# dec output data => Same as dec input data but excluding '/t' of our target sentence

def bagofcharacters(input_texts, target_texts):
    # initialize enc, dec input and target data
    enc_inp = []
    dec_inp = []
    dec_tar = []

    # padding variable with first character as 1 rest as 0
    pad_enc = [1] + [0] * (len(input_characters) - 1)
    pad_dec = [0] * (len(target_characters))
    pad_dec[2] = 1

    # Count Vectorizer for One-Hot Encoding
    # binary = True, sets non-zero counts as 1 (Binary over integer counts)
    # stop_words = None, will not use any stop words
    # overriding tokenizing step, if analyser = "word" or "char"
    cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')

    for i, (input_t, target_t) in enumerate(zip(input_texts, target_texts)):
        # fit the input characters, zip function to map input_texts, target_texts
        # returns zip object which is iterator of tuples

        cv_inp = cv.fit(input_characters)
        # transform the input text using CV
        enc_inp.append(cv_inp.transform(list(input_t)).toarray().tolist())

        cv_tar = cv.fit(target_characters)
        dec_inp.append(cv_tar.transform(list(target_t)).toarray().tolist())

        # dec tar excludes '/t' which differs from dec inp
        dec_tar.append(cv_tar.transform(list(target_t)[1:]).toarray().tolist())

        # We need same length of input data as the maximum so, we add extra array of 0's, repeat the same for target too
        # adding padding variable if len(input_text)<max_input_length
        if len(input_t) < max_input_length:
            for _ in range(max_input_length - len(input_t)):
                enc_inp[i].append(pad_enc)
        if len(target_t) < max_target_length:
            for _ in range(max_target_length - len(target_t)):
                dec_inp[i].append(pad_dec)
        if (len(target_t) - 1) < max_target_length:
            for _ in range(max_target_length - len(target_t) + 1):
                dec_tar[i].append(pad_dec)

    # Converting list to numpy array of datatype float32
    enc_inp = np.array(enc_inp, dtype="float32")
    dec_inp = np.array(dec_inp, dtype="float32")
    dec_tar = np.array(dec_tar, dtype="float32")

    return enc_inp, dec_inp, dec_tar


# 4. Build te training model
# LSTM(Long Short Term Memory) is a type of Recurrent Neural Network that is used when RNN fails


# ENCODER
# create input object whose shape is eq to total number of encoder input characters
enc_inputs = Input(shape=(None, num_en_chars))

# create LSTM with hidden dimension of 256
# ONLY return state = True as we don't want output sequence
encoder = LSTM(256, return_state=True)

# discard encoder output and store hidden and cell state
enc_outputs, hid_state, cell_state = encoder(enc_inputs)
enc_states = [hid_state, cell_state]

# DECODER
# We shall use Softmax Activation function and Dense layer for output
# create input object whose shape is eq to total number of target characters
dec_inputs = Input(shape=(None, num_dec_chars))

# create LSTM with hidden dimension of 256
# return state as well as return sequences is True as we need full output sequence(text), states
decoder = LSTM(256, return_sequences=True, return_state=True)

# initialize the decoder model with states on encoder
dec_outputs, _, _ = decoder(dec_inputs, initial_state=enc_states)

# Dense layer for Output with shape of total number of decoder characters
dec_dense = Dense(num_dec_chars, activation="softmax")
dec_outputs = dec_dense(dec_outputs)

# 5. Train the Model
"""To train the model we fit ('encoder input and decoder input') to turn into ('decoder target data') using 'Adam' optimizer with validation split of 0.2 and epoch of 200
in batch size of 64"""
# batch size gives the number of training examples to be used in one iteration
# Adam is replacement optimization algo for stochastic GD for training Deep Learning models(adaptive optimizers)
# Adam has faster computation time, and reqs fewer parameters for tuning
# create Mode and store all variables
model = Model([enc_inputs, dec_inputs], dec_outputs)

pickle.dump(
    {'input_characters': input_characters, 'target_characters': target_characters, 'max_input_length': max_input_length,
     'max_target_length': max_target_length, 'num_encoder_characters': num_en_chars,
     'num_decoder_characters': num_dec_chars}, open("training_data.pkl", "wb"))

# load the data and train the model
enc_inp_data, dec_inp_data, dec_tar_data = bagofcharacters(input_texts, target_texts)
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
# Categorical cross-entropy is used when true labels are one-hot encoded
# Sparse categorical cross-entropy is used in case of numerical labels

model.fit([enc_inp_data, dec_inp_data], dec_tar_data, batch_size=6, epochs=200, validation_split=0.2)

# save the model
model.save("s2s")

# After the model gets trained we'll get directory as 's2s' with 'saved_model.pb' which includes optimizer,
# losses and accuracy metrics. The weights are saved in variables/directory
"""Summary of our model contains:
 1) Layers that we have used for our model 
 2) Output Shape which shows dimensions or shapes of our layers 
 3) The number of parameters for every layer is the total number of output size, 
 i.e. number of neurons associated with the total number of input weights and one weight of connection with bias so basically,99988sssssssssa   HNMHG,  `1345678/.,vcdsa

N (number of parameters) = (number of neurons) * (number of inputs + 1)
Number of parameters for our dense layer(output layer) will be number of decoder characters present i.e 67 associated with number of input weights i.e 256 and one weight connection with bias

N = 67 * (256 + 1) = 17219"""



# summary and model plot
import keras
from tensorflow.keras.utils import plot_model

model = keras.models.load_model("s2s")
model.summary()

plot_model(model, to_file="Model_plot.png", show_shapes=True, show_layer_names=True)
