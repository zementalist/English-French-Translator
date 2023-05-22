import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, TimeDistributed, LSTM, Embedding, Bidirectional
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_heads = 3  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
embedding_dim = 200
maxlen = 55

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# class TokenAndPositionEmbedding(tf.keras.layers.Layer):

#     def __init__(self, maxlen=55, vocab_size=13582, embed_dim=200, **kwargs):
#         self.maxlen = maxlen
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.token_emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
#         self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)
#         super().__init__(**kwargs)
#     # def get_config(self):
#     #     config = super().get_config().copy()
#     #     print("dude")
#     #     print(config)
#     #     config.update({
#     #         'maxlen': self.maxlen,
#     #         'vocab_size': self.vocab_size,
#     #         'embed_dim': self.embed_dim,
#     #         # 'token_emb': self.token_emb,
#     #         # 'pos_emb': self.pos_emb,
#     #     })
#     #     return config
    
#     @classmethod
#     def from_config(cls, config):
#         print("WWE ALL DUDES")
#         print(config)
#         pos_emb = config['pos_emb']
#         token_emb = config['token_emb']
#         del config['pos_emb']
#         del config['token_emb']
#         instance = cls(
#             maxlen=55,
#             vocab_size=13582,
#             embed_dim=200,
#             **config
#         )
#         instance.token_emb = token_emb#config['token_emb']
#         instance.pos_emb = pos_emb#config['pos_emb']
#         return instance

#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions

# class TransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, embed_dim=200, num_heads=3, ff_dim=32, rate=0.1,**kwargs):
        
#         self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = tf.keras.Sequential(
#             [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
#         )
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)
#         super(TransformerBlock, self).__init__(**kwargs)
#     # def get_config(self):

#     #     config = super().get_config().copy()
#     #     config.update({
#     #         'att': self.att,
#     #         'ffn': self.ffn,
#     #         'layernorm1': self.layernorm1,
#     #         'layernorm2': self.layernorm2,
#     #         'dropout1': self.dropout1,
#     #         'dropout2': self.dropout2,
#     #     })
#     #     return config

#     @classmethod
#     def from_config(cls, config):
#         print("WWE ALL DUDES")
#         print(config)
#         dropout2 = config['dropout2']
#         att = config['att']
#         ffn = config['ffn']
#         layernorm1 = config['layernorm1']
#         layernorm2 = config['layernorm2']
#         dropout1 = config['dropout1']
#         del config['att']
#         del config['ffn']
#         del config['layernorm1']
#         del config['layernorm2']
#         del config['dropout1']
#         del config['dropout2']
#         instance = cls(
#             num_heads=3,
#             ff_dim=32,
#             embed_dim=200,
#             rate=0.1,
#             name="transformer_block",
#             trainable=True,
#             dtype="float32"
#             # **config
#         )
#         instance.att = att#config['token_emb']
#         instance.ffn = ffn#config['pos_emb']
#         instance.layernorm1 = layernorm1#config['pos_emb']
#         instance.layernorm2 = layernorm2#config['pos_emb']
#         instance.dropout1 = dropout1#config['pos_emb']
#         instance.dropout2 = dropout2#config['pos_emb']
#         return instance
#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)

def create_model(input_vocab_size, output_vocab_size):
    adam = Adam(learning_rate=0.003)


    inputs = tf.keras.layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, input_vocab_size, embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = TimeDistributed(Dense(256, activation="relu"))(x)
    outputs = TimeDistributed(Dense(output_vocab_size, activation="softmax"))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
    return model

def get_translation_model(input_vocab_size, output_vocab_size):
    # model = tf.keras.models.load_model("MachineTrans.h5", custom_objects={
    #     "TransformerBlock":TransformerBlock,
    #     "TokenAndPositionEmbedding":TokenAndPositionEmbedding
    # })
    model = create_model(input_vocab_size, output_vocab_size)
    model.load_weights("./models/mtt_weights.h5")
    return model


def translate_text(samples, model, en_tokenizer, fr_tokenizer):
    result = []
    if type(samples) == str:
        samples = [samples]
    if type(samples) == list:
        for sample in samples:
            pred = model.predict([pad_sequences(en_tokenizer.texts_to_sequences([sample]), maxlen=maxlen, padding='post', truncating='post')])[0].argmax(axis=1)
            output_text = fr_tokenizer.sequences_to_texts([pred])[0]
            print("EN: " + sample)
            print("FR: " + output_text)
            print()
            result.append(output_text)
    else:
        raise TypeError("Expected argument (samples) to be of type str or list")
    return result





def create_lr_model(vocabSize):
    model = Sequential()
    model.add(Embedding(vocabSize, 200, input_length=48))
    model.add(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_lr_model(tokenizer):
    vocabSize = len(tokenizer.word_index) + 1
    model = create_lr_model(vocabSize)
    model.load_weights("./models/learning_styles_weights.h5")
    return model

def id_to_learning_style(id):
    mapping = {
        0: "Auditory",
        1: "Kinesthetic",
        2: "Visual"
    }
    return mapping[id]

def recognize_learning_style(samples, model, tokenizer):
    result = []
    if type(samples) == str:
        samples = [samples]
    if type(samples) == list:
        for sample in samples:
            pred = model.predict([pad_sequences(tokenizer.texts_to_sequences([sample]), maxlen=48, truncating='pre')]).argmax(axis=-1)[0]
            output_text = id_to_learning_style(pred)
            print("Text: " + sample)
            print("Learning Style: " + output_text)
            print()
            result.append(output_text)
    else:
        raise TypeError("Expected argument (samples) to be of type str or list")
    return result