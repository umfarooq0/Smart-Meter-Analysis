
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.convlayer1 = layers.Conv2D(24,(1, 1), activation="relu", strides=1, padding="same")
        self.flat = layers.Flatten()
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.convlayer1(inputs)
        x = self.flat(x)
        x = self.dense_proj(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.shaping = layers.Reshape((673, 48, 1))
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.convlayer1T = layers.Conv2DTranspose(24,(1, 1), activation="relu", strides=1, padding="same")
    

    def call(self, inputs):
        x = self.shaping(inputs)
        x = self.dense_proj(x)
        x = self.convlayer1T(x)
        return x


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed




df_array = pd.read_csv('/home/usman/Documents/Smart-Meter-analysis/total_hhblock.csv')
df_array = df_array.iloc[:,1:]
LCLid_ = df_array.iloc[:,0]
#df_array = df_array.iloc[:,2:]


test_set = df_array[df_array.LCLid ==LCLid_[0] ]
test_set = test_set.dropna()

test_set_ = test_set.to_numpy()

scaler = MinMaxScaler()
test_set_ = scaler.fit_transform(test_set_[:,2:])

input_shape = test_set_.shape[1]
latent_dim = 5
layer1 = 96
epochs = 10
batch_size = 128

vae = VariationalAutoEncoder(input_shape, layer1, latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
test_vae = vae.fit(test_set_, test_set_, epochs=epochs, batch_size=batch_size)


def train_vae(df,idno,input_shape,layer1,latent_dim,epochs,batch_size):
    
    test_set = df[df.LCLid ==idno]
    test_set = test_set.dropna()

    test_set_ = test_set.to_numpy()

    scaler = MinMaxScaler()
    test_set_ = scaler.fit_transform(test_set_[:,2:])


    vae = VariationalAutoEncoder(input_shape, layer1, latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(test_set_, test_set_, epochs=epochs, batch_size=batch_size)
    return vae.get_weights()[3]


test_ids = LCLid_[:10]
vae_features = [train_vae(df_array,x,input_shape,layer1,latent_dim,3,8) for x in test_ids]

features_ = pd.DataFrame(vae_features)
features_.to_csv('vae_features_.csv')