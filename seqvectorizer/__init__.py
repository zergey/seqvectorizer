# -*- coding: utf-8 -*-
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.models import load_model
from sklearn.externals import joblib
from sklearn import preprocessing


class NotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting.
    """


class SeqVectorizer:
    """sequence into dense vectors of fixed size

    Parameters
    ----------

    n_dims : int, optional, default: 8
        The number of dimensions for embedding vectors.

    max_iter : int, default: 300
        Maximum number of iterations of the training algorithm.
    batch_size : int, default: 256
        Batch size for autoencoder training
    max_seq_len : int, default: 0
        Maximum len of sequence. Sequences longer than limit are truncated. 0 for no limit
    encoder_optimizer : str, default: 'rmsprop'
        Optimization algorythm for autoencoder training
    encoder_loss : str, default: 'categorical_crossentropy'
        Loss function for autoencoder training
    start_token : str, default: "[START]"
        Token to be added to the sequence beginning, set to None if not required
    end_token : str, default: "[END]"
        Token to be added to the sequence beginning, set to None if not required
    encoder_validation_split : float, default: 0.3
        Validation split for autoencoder training
    verbose : int, default 0
        Verbosity mode for autoencoder training. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    Attributes
    ----------

    """

    def __init__(self,
                 n_dims=8,
                 max_iter=300,
                 batch_size=256,
                 max_seq_len=0,
                 encoder_optimizer="rmsprop",
                 encoder_loss='categorical_crossentropy',
                 start_token="[START]",
                 end_token="[END]",
                 encoder_validation_split=0.3,
                 verbose=0):
        self._n_dims = n_dims
        self._max_iter = max_iter
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._verbose = verbose
        self._encoder_loss = encoder_loss
        self._encoder_optimizer = encoder_optimizer
        self._start_token = start_token
        self._end_token = end_token
        self._encoder_validation_split = encoder_validation_split
        self._model_ready = False
        self._model_history = None
        self._encoder_model = None
        self._tokens_encoder = None
        self._num_tokens = 0

    def _check_is_fitted(self):
        msg = ("This instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")
        if not self._model_ready:
            raise NotFittedError(msg)

    def fit(self, X, y=None):
        """Train sequence encoder.

        Parameters
        ----------
        X : Training instances to embed.

        y : Ignored

        """
        self._fit_token_encoder(X)
        decoder_input_data, decoder_target_data, encoder_input_data = self._create_ae_matrix(X)
        self._train_ae(decoder_input_data, decoder_target_data, encoder_input_data)
        return self

    def _train_ae(self, decoder_input_data, decoder_target_data, encoder_input_data):
        encoder_inputs = Input(shape=(None, self._num_tokens))
        encoder = LSTM(self._n_dims,
                       return_state=True,
                       name="enc_lstm")
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        decoder_inputs = Input(shape=(None, self._num_tokens))
        decoder_lstm = LSTM(self._n_dims,
                            return_sequences=True,
                            return_state=True,
                            name="dec_lstm")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self._num_tokens,
                              activation='softmax',
                              name="dec_dense")
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=self._encoder_optimizer,
                      loss=self._encoder_loss)
        self._model_history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                        batch_size=self._batch_size,
                                        epochs=self._max_iter,
                                        validation_split=self._encoder_validation_split,
                                        verbose=self._verbose,
                                        )
        self.encoder_model = Model(encoder_inputs, encoder_states)
        self._model_ready = True

    def _create_ae_matrix(self, X):
        num_samples = len(X)
        max_input_seq_length = max([len(s) for s in X])
        if self._max_seq_len:
            max_input_seq_length = min(self._max_seq_len, max_input_seq_length)
        max_seq_length = max_input_seq_length
        max_seq_length += 1 if self._start_token else 0
        max_seq_length += 1 if self._end_token else 0
        dtype = 'int8'
        encoder_input_data = np.zeros(
            (num_samples, max_seq_length, self._num_tokens),
            dtype=dtype)
        decoder_input_data = np.zeros(
            (num_samples, max_seq_length, self._num_tokens),
            dtype=dtype)
        decoder_target_data = np.zeros(
            (num_samples, max_seq_length, self._num_tokens),
            dtype=dtype)
        for i, input_seq in enumerate(X):
            token_seq = []
            if self._start_token:
                token_seq.append(self._start_token)
            if len(input_seq) <= max_seq_length:
                token_seq.extend(input_seq)
            else:
                token_seq.extend(input_seq[:max_input_seq_length])
            if self._end_token:
                token_seq.append(self._end_token)
            for t, token_id in enumerate(self._tokens_encoder.transform(token_seq)):
                encoder_input_data[i, t, token_id] = 1.
                decoder_input_data[i, t, token_id] = 1.
                if not t:
                    continue
                decoder_target_data[i, t - 1, token_id] = 1.
                if t >= max_seq_length:
                    break
        return decoder_input_data, decoder_target_data, encoder_input_data

    def _fit_token_encoder(self, X):
        tokens = set()
        if self._start_token:
            tokens.add(self._start_token)
        if self._end_token:
            tokens.add(self._end_token)
        for x in X:
            _ = [tokens.add(t) for t in x]
        tokens = list(tokens)
        self._tokens_encoder = preprocessing.LabelEncoder().fit(tokens)
        self._num_tokens = len(tokens)

    def fit_transform(self, X, y=None):
        """Train autoencoder and embed X to dense vectors.

        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : Training instances to embed

        y : Ignored

        Returns
        -------
        """
        self._fit_token_encoder(X)
        decoder_input_data, decoder_target_data, encoder_input_data = self._create_ae_matrix(X)
        self._train_ae(decoder_input_data, decoder_target_data, encoder_input_data)

        return self._transform(encoder_input_data)

    def _transform(self, encoder_input_data):
        seq_value = self.encoder_model.predict(encoder_input_data)
        encoded = np.concatenate(seq_value, axis=1)
        return encoded

    def transform(self, X):
        """embed X to dense vectors.

        Parameters
        ----------
        X : instances to embed

        Returns
        -------
        X_new : array, shape [n_samples, n_dims]
            X transformed in the new space.
        """
        self._check_is_fitted()
        _, _, encoder_input_data = self._create_ae_matrix(X)
        return self._transform(encoder_input_data)

    def score(self):
        history = self._model_history.history
        return {x:history[x][-1] for x in history}

    def save(self, sequence_encoder_path, token_encoder_path):
        self._encoder_model.save(sequence_encoder_path)
        joblib.dump(self._tokens_encoder, token_encoder_path)

    def load(self, sequence_encoder_path, token_encoder_path):
        self._encoder_model = load_model(sequence_encoder_path)
        self._tokens_encoder = joblib.load(token_encoder_path)
        self._model_ready = True
