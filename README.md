# eng-french-translator
This app translates English input to French

Model developed using LSTM Recurrent Neural Networks implemented using Keras, Tensorflow APIs.
Employed many-to-many encoder-decoder sequence model for translation. Adam optimizer with batch size of 64 and epoch of 200.

*Teacher Forcing Algorithm (TFA):* The TFA network model uses ground truth input rather than output from the previous model.

**Project Structure:**

* **eng-french.txt:** Dataset used for the project that contatins huge set of English texts and their corresponding French texts.

* **langTraining.py:** Core of the project where we build and train the model.

* **training_data.pkl:** Pickle file that contains serialised Python objects like lists of characters, text of input, target data, variables, functions in binary format.

* **s2s:** Directory containing optimizers, metrics, weights of our trained model. It contains the saved model, variables and assets.

* **LangTransGUI.py:** GUI file which implements Tkinter for loading the trained model and creating an interactive website where input data is translated.

* **Images:** This directory consists of notes on LSTM and Neural Networks along with the Output Screenshots


**Steps to develop Language Translator App:**

1. Import Libraries and initialize variables.

2. Parse the dataset file

3. One Hot Encoding (Vectorization)

4. Build the training model

5. Train the Model

6. Create GUI for prediction

7. Inference(Sampling) model and prediction

8. Run Language Translation Code File


