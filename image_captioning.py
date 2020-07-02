# Importing necessary libraries....
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Getting the list of images....
images_list = os.listdir('dataset/flickr30k_images/flickr30k_images/')
print(len(images_list))

image_size = (224, 224)
num_channels = 3

# Taking only the first 5000 images.....
sample_size = 5000
sample_images_list = images_list[:sample_size]

# Reading the images from the images source path....
images = []
for img_name in sample_images_list:
    images.append(plt.imread('dataset/flickr30k_images/flickr30k_images/' + img_name))

images = np.array(images)

# Reshaping the images for compatibilty with VGG16's input.....
for i in range(images.shape[0]):
    images[i] = cv2.resize(images[i], image_size)
    images[i] = images[i].reshape(1, image_size[0], image_size[1], num_channels)

# Vertically stacking the images....
images = np.vstack(images[:])
print(images.shape)  # shape is (5000,224,224,3)

plt.imshow(images[12])
plt.show()

# Reading the image captions from results.csv file.....
images_caption = pd.read_csv('dataset/flickr30k_images/results.csv', delimiter='|')
images_caption.columns = ['image_name', 'comment_number', 'comment']
images_caption.head()

# Function to get the captions as a list....
def get_captions(images_list, images_caption):
    captions_list = []
    for img_name in images_list:
        captions_list.append(images_caption[images_caption['image_name'] == img_name]['comment'].iat[0])
    return captions_list

captions = np.array(get_captions(sample_images_list, images_caption))
print("Total Captions :", len(captions))

# To get the captions as a list of lists(all captions for an image in one list).....
temp = images_caption.groupby('image_name')['comment'].apply(list).reset_index(name='comment')
df = pd.DataFrame(temp, columns= ['comment'])
captions_listoflist = df.values.tolist()
captions_listoflist = captions_listoflist[:5000]
captions_listoflist = np.array(captions_listoflist).reshape(5000,5)

# Pre-trained VGG16 model as an encoder......
image_model = VGG16(include_top=True, weights='imagenet')
image_model.summary()

# Input and output for the new model.....
new_input = image_model.input
hidden_layer = image_model.get_layer('fc2')

# transfer values size for input to the initial states of decoder model....
transfer_values_size = K.int_shape(hidden_layer.output)[1]
print(transfer_values_size)

# Modified encoder model for getting the tranfer values of the images.....
image_features_extract_model = tf.keras.Model(inputs=new_input,outputs=hidden_layer.output)
image_features_extract_model.summary()
transfer_values = image_features_extract_model.predict(images,batch_size=32,verbose=1)


# Adding a start and end word to all the captions.....
mark_start = 'ssss '
mark_end = ' eeee'

def mark_captions(captions_listoflist):
    captions_marked =  [[mark_start + caption + mark_end
                        for caption in caption_list] for caption_list in captions_listoflist]
    return captions_marked

captions_train = mark_captions(captions_listoflist)
print(captions_train[0])


# Flattening the list of lists....
captions_train_flatten = [caption for captions_list in captions_train for caption in captions_list]
print(captions_train_flatten[0])


# Tokenizer class with various methods and Properties.....
# vocab size....
num_words = 10000
class Tokenizer_Prop(Tokenizer):
    def __init__(self,text,num_words=None):
        Tokenizer.__init__(self,num_words=num_words)
        self.fit_on_texts(text)
        self.index_to_word = dict(zip(self.word_index.values(),self.word_index.keys()))

    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        text = " ".join(words)
        return text
    
    def captions_to_tokens(self, captions_listlist):
        tokens = [self.texts_to_sequences(captions_list) for captions_list in captions_listoflist]
        return tokens

# Creating an object of the Tokenizer_Prop class....
tokenizer = Tokenizer_Prop(text=captions_train_flatten,num_words=num_words)

# token for the start word....
token_start = tokenizer.word_index[mark_start.strip()]
print(token_start)

# Token for the end word....
token_end = tokenizer.word_index[mark_end.strip()]
print(token_end)

# Converting the captions to tokens.....
token_train = tokenizer.captions_to_tokens(captions_train)
print(token_train[0])

def get_random_caption_tokens(idx):
    result = []
    for i in idx:
        j = np.random.choice(len(token_train[i]))
        tokens = token_train[i][j]
        result.append(tokens)
    return result

def batch_generator(batch_size):
    while True:
        # Get a list of random indices for images in the training-set....
        idx = np.random.randint(len(captions_train), size=batch_size)
        transfer_values_temp = transfer_values[idx]

        # Select one of the 5 captions for the selected image at random and get the associated sequence of integer-tokens...
        tokens = get_random_caption_tokens(idx)

        # Count the number of tokens in all these token-sequences.
        num_tokens = [len(t) for t in tokens]
        
        # Max number of tokens.
        max_tokens = np.max(num_tokens)
        
        tokens_padded = pad_sequences(tokens,maxlen=max_tokens, padding='post',truncating='post')

        # The decoder-part of the neural network will try to map the token-sequences to themselves shifted one time-step....
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # Dict for the input-data. Because we have several inputs, we use a named dict to ensure that the data is assigned correctly....
        x_data = \
        {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values_temp
        }

        # Dict for the output-data.
        y_data = \
        {
            'decoder_output': decoder_output_data
        }
        
        yield (x_data, y_data)
        
batch_size = 128

generator = batch_generator(batch_size=batch_size)
batch = next(generator)
batch_x = batch[0]
batch_y = batch[1]
print(batch_x['transfer_values_input'][0])

num_captions_train = [len(captions) for captions in captions_listoflist]
total_num_captions_train = np.sum(num_captions_train)
steps_per_epoch = int(total_num_captions_train / batch_size)
print(steps_per_epoch)

state_size = 512

embedding_size = 128

transfer_values_input = Input(shape=(transfer_values_size,),name='transfer_values_input')

decoder_transfer_map = Dense(state_size,activation='tanh',name='decoder_transfer_map')

decoder_input = Input(shape=(None, ), name='decoder_input')

decoder_embedding = Embedding(input_dim=num_words,output_dim=embedding_size,name='decoder_embedding')


decoder_gru1 = GRU(state_size, name='decoder_gru1',return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',return_sequences=True)

decoder_dense = Dense(num_words,activation='softmax',name='decoder_output')

def connect_decoder(transfer_values):
    # Map the transfer-values so the dimensionality matches the internal state of the GRU layers....
    initial_state = decoder_transfer_map(transfer_values)

    # Start the decoder-network with its input-layer.....
    net = decoder_input
    
    # Connect the embedding-layer....
    net = decoder_embedding(net)
    
    # Connect all the GRU layers....
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to one-hot encoded arrays....
    decoder_output = decoder_dense(net)
    
    return decoder_output

decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],outputs=[decoder_output])

decoder_model.summary()

from keras.utils import plot_model
plot_model(decoder_model, to_file='image_model.png')

decoder_model.compile(optimizer=RMSprop(lr=1e-3),loss='sparse_categorical_crossentropy')

decoder_model.fit(x=generator,steps_per_epoch=steps_per_epoch,epochs=1)

# Generating a caption for the given image....

def generate_caption(max_tokens=50):
    # Load and resize the image
    test_image = plt.imread('dataset/flickr30k_images/flickr30k_images/'+ images_list[10104])
    test_image = cv2.resize(test_image, image_size)
    # Expand the 3-dim numpy array to 4-dim as the image-model expects a whole batch as input....
    image_batch = np.expand_dims(test_image, axis=0)

    # Process the image with the pre-trained image-model to get the transfer-values....
    transfer_values_test = image_features_extract_model.predict(image_batch)
    
    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder with the last token that was sampled....
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'transfer_values_input': transfer_values_test,
            'decoder_input': decoder_input_data
        }
        
        # Input this data to the decoder and get the predicted output....
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array....
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.....
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.....
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.....
        output_text += " " + sampled_word

        # Increment the token-counter.....
        count_tokens += 1

    # This is the sequence of tokens output by the decoder....
    output_tokens = decoder_input_data[0]

    # Plot the image.....
    plt.imshow(test_image)
    plt.show()
    
    # Print the predicted caption....
    print("Predicted caption:")
    print(output_text)
    print()

generate_caption()
