from tkinter import *
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Input, LSTM, Dense

# tkinter is the only framework built into the Python standard library to create GUIs
# 6. Creating GUI for prediction

BG_GRAY = "#ABB2B9"
BG_COLOR = "#000"
TEXT_COLOR = "#FFF"
FONT = "Melvetica 14"
FONT_BOLD = "Melvectica 13 bold"

cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')


class LangTrans:
    def __init__(self):
        # initialize tkinter window and load the file
        self.window = Tk()
        # It helps to display the root window and manages all other components of tkinter application
        self.main_window()
        self.datafile()

    # Load all the variables from file using pickle module
    def datafile(self):
        datafile = pickle.load(open("training_data.pkl", 'rb'))
        self.input_characters = datafile['input_characters']
        self.target_characters = datafile['target_characters']

        self.max_inp_len = datafile['max_input_length']
        self.max_tar_len = datafile['max_target_length']

        self.num_inp_chars = datafile['num_en_chars']
        self.num_tar_chars = datafile['num_dec_chars']

        self.load_model()

    # Create main window for lang translation: create scrollbar, title, text widget for GUI
    def main_window(self):
        self.window.title("Language Translator")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=520, height=520, bg=BG_COLOR)

        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome to English-French Translator", font=FONT_BOLD,
                           pady=10)
        head_label.place(relwidth=1)

        line = Label(self.window, width=450, bg=BG_COLOR)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # create text widget where input and output text will be displayed
        self.text_widget = Text(self.window, width=20, height=2, bg="#FFF", fg="#000", font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # create scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # create bottom label where text widget will be placed
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # this is for user to put english input
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.788, relheight=0.06, rely=0.008, relx=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self.on_enter)

        # send button which calls on_enter function to send the text
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=8, bg="#FFF",
                             command=lambda: self.on_enter(None))
        send_button.place(relx=0.80, rely=0.008, relheight=0.06, relwidth=0.20)

    # 7. Inference(Sampling) model and prediction Load the saved model and construct encoder and decoder.We will get
    # the inputs from the saved model and LSTM to get the hidden and cell state of the encoder which is required to
    # create the encoder model
    def load_model(self):
        # Sampling mode
        # Load the model
        model = models.load_model("s2s")

        # construct encoder model from output of the second layer'
        # discard the encoder output and store only the states
        enc_outputs, hid_enc_state, cell_enc_state = model.layers[2].output

        # add input object and state from the layer
        self.enc_model = Model(model.input[0], [hid_enc_state, cell_enc_state])

        # for decoder, we take 2nd input and create an input object for hidden as well as for cell state of shape(256,)
        # which is latent(hidden) dimension of layer. we'll run one step of decoder with this initial state and a start
        # of text character after that our output will be next character of text.

        # create Input object for hidden and cell state for decoder
        dec_state_input_h = Input(shape=(256,), name='input_3')
        dec_state_input_c = Input(shape=(256,), name='input_4')
        dec_states_inputs = [dec_state_input_h, dec_state_input_c]

        # add input from encoder output and initialize with the states
        dec_lstm = model.layers[3]
        dec_outputs, dec_hid, dec_cell = dec_lstm(model.input[1], initial_state=dec_states_inputs)

        dec_states = [dec_hid, dec_cell]
        dec_dense = model.layers[4]
        dec_outputs = dec_dense(dec_outputs)

        # create model with input of decoder state input and encoder input and decoder output with decoder states
        self.dec_model = Model([model.input[1]] + dec_states_inputs, [dec_outputs] + dec_states)

    # Encode the input sequence as state vectors, create empty array of target sequence of length 1 and generate
    # start character('\t') of every pair to be 1. Use thi state value along with input sequence to predict output index
    # Use reverse character index to get the character from output index and append to the decoded sequence

    def decode_sequence(self, input_seq):
        # create dictionary with key as index and value as characters
        reverse_target_index = dict(enumerate(self.target_characters))

        # get states from user input seq
        states_value = self.enc_model.predict(input_seq)

        # fit target characters and initialize every first character to be 1 which is '\t'
        # Generate empty target sequence of length 1
        co = cv.fit(self.target_characters)
        target_seq = np.array([co.transform(list('\t')).toarray().tolist()], dtype="float32")

        # if iteration reaches end of text then it will stop the iteration
        stop_condition = False
        # append every predicted character in decoded sentence
        decoded_sentence = ""

        while not stop_condition:
            # get predicted output and discard hidden and cell state
            output_chars, h, c = self.dec_model.predict([target_seq] + states_value)

            # get the index and from dictionary get the character
            char_index = np.argmax(output_chars[0, -1, :])
            text_char = reverse_target_index[char_index]
            decoded_sentence += text_char

            """ For ey index, put 1 to that index of our target array. For next iteration, our target sequence will be
             having a vector of previous character. Iterate until our character is equal to the last character
             (stop character) or max len of the target text"""

            if text_char == "\n" or len(decoded_sentence) > self.max_tar_len:
                stop_condition = True
            # update target sequence to the current character index
            target_seq = np.zeros((1, 1, self.num_tar_chars))
            target_seq[0, 0, char_index] = 1.0
            states_value = [h, c]
        # return the decoded sentence
        return decoded_sentence

    def on_enter(self, event):
        # get user query and bot response
        msg = self.msg_entry.get()
        self.my_msg(msg, "English")
        self.decoded_output(msg, "French Translation")

    # Get (English) input text from user and pass it to bagofcharacters for one-hot-encoding process.
    # Then pass encoded vector into decode_sequence() for decoding to (French) text

    def bagofcharacters(self, input_text):
        cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')
        enc_input_data = []
        pad_en = [1] + [0] * (len(self.input_characters) - 1)

        cv_inp = cv.fit(self.input_characters)
        enc_input_data.append(cv_inp.transform(list(input_text)).toarray().tolist())

        if len(input_text) < self.max_inp_len:
            for _ in range(self.max_inp_len - len(input_text)):
                enc_input_data[0].append(pad_en)

        return np.array(enc_input_data, dtype="float32")

    def decoded_output(self, msg, sender):
        self.text_widget.configure(state=NORMAL)
        enc_input_data = self.bagofcharacters(msg.lower() + ".")
        self.text_widget.insert(END, str(sender) + " : " + self.decode_sequence(enc_input_data) + "\n\n")
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)

    def my_msg(self, msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0, END)
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, str(sender) + " : " + str(msg) + "\n")
        self.text_widget.configure(state=DISABLED)

    # run window
    def run(self):
        self.window.mainloop()


# run the file
if __name__ == "__main__":
    LT = LangTrans()
    LT.run()
