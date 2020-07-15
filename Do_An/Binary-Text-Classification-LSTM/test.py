import tensorflow as tf
import numpy as np
import buid_dataset
from tensorflow.contrib import rnn


print("Data set is building...")
training_data, train_paragraph_label, train_document_label, train_document_num_paragraph, vocabulary_size, num_words = buid_dataset.run("Data/processed-training-data/","Data/training-class")
testing_data, test_paragraph_label, test_document_label, test_document_num_paragraph, _,  _ = buid_dataset.run("Data/processed-test-data/","Data/test-class", num_words = num_words)
num_classes = 1
timesteps = len(training_data[0])
embedding_size = 300
num_hidden = 250
batch_size = 200
iteration = 30
accuracy_train = 0
accuracy_test = 0


def metrics(y, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for index, element_y in enumerate(y):
        if element_y == 1:
            if y_pred[index] == 0:
                fn+= 1
            else:
                tp+= 1
        else:
            if y_pred[index] == 0:
                tn+= 1
            else:
                fp+= 1
    return tp, tn, fp, fn


def majority_voting(document_num_paragraph, y_pred):
    document_vote = []
    offset = 0
    for x1 in document_num_paragraph:
        positive_vote = 0
        for x2 in range(offset, offset + x1):
            if y_pred[x2] == 1:
                positive_vote += 1
        if positive_vote / x1 > 0.5:
            document_vote.append(1)
        else:
            document_vote.append(0)
        offset = x1

    return document_vote

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
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(prediction)), Y), tf.float32))
init = tf.global_variables_initializer()
print("Testing process started...")
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "Model/model.ckpt")
    last_batch = len(training_data) % batch_size
    training_steps = (len(training_data) // batch_size) + 1
    train_paragraph_label_pred_total = []
    test_paragraph_label_pred_total = []
    for step in range(training_steps):
        X_batch = training_data[(step * batch_size):((step + 1) * batch_size)]
        Y_batch = train_paragraph_label[(step * batch_size):((step + 1) * batch_size)]
        if len(X_batch) < batch_size:
            X_batch = np.array(X_batch)
            X_batch = X_batch.reshape(last_batch, timesteps)
            Y_batch = np.array(Y_batch)
            Y_batch = Y_batch.reshape(last_batch, num_classes)
        else:
            X_batch = np.array(X_batch)
            X_batch = X_batch.reshape(batch_size, timesteps)
            Y_batch = np.array(Y_batch)
            Y_batch = Y_batch.reshape(batch_size, num_classes)
        acc, train_paragraph_label_pred = sess.run([accuracy,tf.round(tf.sigmoid(prediction))], feed_dict={X: X_batch, Y: Y_batch})
        train_paragraph_label_pred_total = np.append(train_paragraph_label_pred_total, train_paragraph_label_pred)
        accuracy_train += acc
    accuracy_train /= training_steps
    train_document_label_pred = majority_voting(train_document_num_paragraph, train_paragraph_label_pred_total)
    tp, tn, fp, fn = metrics(train_document_label, train_document_label_pred)
    #print("----- Train results ----")
    #print ("True Positive = " + str(tp))
    #print ("False Positive  = " + str(fp))
    #print ("True Negative = " + str(tn))
    #print ("False Negative = " + str(fn))

    last_batch = len(testing_data) % batch_size
    testing_steps = (len(testing_data) // batch_size) + 1
    for step in range(testing_steps):
        X_batch = testing_data[(step * batch_size):((step + 1) * batch_size)]
        Y_batch = test_paragraph_label[(step * batch_size):((step + 1) * batch_size)]
        if len(X_batch) < batch_size:
            X_batch = np.array(X_batch)
            X_batch = X_batch.reshape(last_batch, timesteps)
            Y_batch = np.array(Y_batch)
            Y_batch = Y_batch.reshape(last_batch, num_classes)
        else:
            X_batch = np.array(X_batch)
            X_batch = X_batch.reshape(batch_size, timesteps)
            Y_batch = np.array(Y_batch)
            Y_batch = Y_batch.reshape(batch_size, num_classes)
        acc, test_paragraph_label_pred = sess.run([accuracy,tf.round(tf.sigmoid(prediction))], feed_dict={X: X_batch, Y: Y_batch})
        test_paragraph_label_pred_total = np.append(test_paragraph_label_pred_total, test_paragraph_label_pred)
        accuracy_test += acc
    accuracy_test /= testing_steps
    test_document_label_pred  = majority_voting(test_document_num_paragraph, test_paragraph_label_pred_total)
    tp, tn, fp, fn = metrics(test_document_label, test_document_label_pred)
    #print("----- Test results ----")
    #print ("True Positive = " + str(tp))
    #print ("False Positive  = " + str(fp))
    #print ("True Negative = " + str(tn))
    #print ("False Negative = " + str(fn))
if tp >0 :
    print("18+")
if fp >0 :
    print("dưới 18+")









