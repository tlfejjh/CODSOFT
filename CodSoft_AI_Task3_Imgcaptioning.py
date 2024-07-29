# prompt: Combine computer vision and natural language processing to build an image captioning AI. Use pre-trained image recognition models like VGG or ResNet to extract features from images, and then use a recurrent neural network (RNN) or transformer-based model to generate captions for those images.

# Install necessary libraries
!pip install tensorflow keras numpy nltk

# Import libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu

# Download necessary NLTK data
nltk.download('punkt')

# Load pre-trained image recognition model (VGG16)
image_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Prepare image data
def preprocess_image(image_path):
  img = load_img(image_path, target_size=(224, 224))
  img_array = img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
  return img_array

# Extract features from images
def extract_features(image_path):
  img_array = preprocess_image(image_path)
  features = image_model.predict(img_array)
  return features

# Prepare text data
def prepare_text_data(captions):
  # Tokenize captions, build vocabulary, etc.
  # ...
  return tokenizer, max_length

# Build image captioning model
def build_captioning_model(vocab_size, max_length):
  inputs1 = Input(shape=(4096,))  # Input for image features
  fe1 = Dense(256, activation='relu')(inputs1)
  inputs2 = Input(shape=(max_length,))  # Input for caption sequence
  se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
  se2 = LSTM(256)(se1)
  decoder1 = tf.keras.layers.add([fe1, se2])
  decoder2 = Dense(256, activation='relu')(decoder1)
  outputs = Dense(vocab_size, activation='softmax')(decoder2)
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)
  return model

# Train the model
# ...

# Generate captions for images
def generate_caption(image_path):
  features = extract_features(image_path)
  # ... (Use the trained model to generate caption)
  return generated_caption

# Evaluate the model using BLEU score
def evaluate_model(actual_captions, generated_captions):
  bleu_score = corpus_bleu(actual_captions, generated_captions)
  return bleu_score



#PART-2

!pip install transformers
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

predict_step(['/content/drive/MyDrive/Home.jpg'])
