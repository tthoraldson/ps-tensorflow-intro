import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation # import animation support

# generate some houses between 1000 and 3500 square feet
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# generate house prices from house size with some noise
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# Plot the points on a cool graph
# plt.plot(house_size, house_price, "bx") # bx = blue x
# plt.xlabel('House Size')
# plt.ylabel('House Price')
# plt.show()

# Data Preperation
# Normalize the array
def normalize(array):
    return (array - array.mean()) / array.std()

# Define number of training examples
num_train_samples = math.floor(num_house * 0.7) # 0.7 = 70%

# Define training Data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# Set up tensorflow place holders
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# Initalize size factors // set to random floats
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# Define operations used for predicting values
# predicted price = (house_size * size_factor) + price_offset
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# Define the loss function (mean squared error)
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2)) / (2*num_train_samples)

# Optimizer Learning Rate
learning_rate = 0.1

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initalize all of the tensorflow variables
init = tf.global_variables_initializer()

# Launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    # set how often to display training progress
    display_every = 2
    num_training_iter = 50

    # keep iterating the training data
    for iteration in range(num_training_iter):

        # fit all training data
        for (x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        # Display current status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("Optimizing Finnished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_cost: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), "\n")

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size in square feet")
    plt.plot(train_house_size, train_price, 'go', label="Training Data")
    #plt.plot(test_house_size, test_house_price, 'mo', label="Testing Data")
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
        (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
        label="Learned Regression")

    plt.legend(loc="upper left")
    plt.show()




# affirmation that you're a good human
print('it workin')
