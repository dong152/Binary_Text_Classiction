import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import buid_dataset
import data_cleaning

# Clean test and training data
print("Data set is cleaning...")
data_cleaning.process(documents_folder="Data/training-data/", processed_documents_folder="Data/processed-training-data/")
data_cleaning.process(documents_folder="Data/test-data/", processed_documents_folder="Data/processed-test-data/")
# Make the training-data set ready to feed into the LSTM algorithm
print("Data set is building...")
training_data, labels, document_labels, document_lookup_table, vocabulary_size, _ = buid_dataset.run("Data/processed-training-data/","Data/training-class", shuffle_data = True)


# Parameters
num_classes = 1
timesteps = len(training_data[0])
embedding_size = 300
num_hidden = 250
batch_size = 200
iteration = 2


# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# tf graph input
X = tf.placeholder(tf.int32, [None, timesteps])
Y = tf.placeholder("float", [None, num_classes])

embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), trainable=True)
inputs = tf.nn.embedding_lookup(embeddings_var, X)


def RNN(inputs, weights, biases):

    # Unstack to get a list of 'timesteps' tensors, each tensor has shape (batch_size, n_input)
    x = tf.unstack(inputs, timesteps, 1)

    # Build a LSTM cell
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get LSTM cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(inputs, weights, biases)
prediction = tf.nn.sigmoid(logits)

# Loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_op)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(prediction)), Y), tf.float32))

# Initialize the variables with default values
init = tf.global_variables_initializer()
print("Training process started...")
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    saver = tf.train.Saver()
    for i in range(iteration):
        last_batch = len(training_data) % batch_size
        training_steps = (len(training_data) // batch_size) + 1
        for step in range(training_steps):
            X_batch = training_data[(step * batch_size) :((step + 1) * batch_size)]
            Y_batch = labels[(step * batch_size) :((step + 1) * batch_size)]
            if len(X_batch) < batch_size:
                X_batch = np.array(X_batch)
                X_batch = X_batch.reshape(last_batch, timesteps)
                Y_batch= np.array(Y_batch)
                Y_batch = Y_batch.reshape(last_batch, num_classes)
            else:
                X_batch = np.array(X_batch)
                X_batch = X_batch.reshape(batch_size, timesteps)
                Y_batch = np.array(Y_batch)
                Y_batch = Y_batch.reshape(batch_size, num_classes)
            _, acc, loss = sess.run([train_op, accuracy, loss_op], feed_dict={X: X_batch, Y: Y_batch})
            print("Step " + str(i) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.2f}".format(acc * 100))
    save_path = saver.save(sess, "Model/model.ckpt")





