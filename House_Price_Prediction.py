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
plt.plot(house_size, house_price, "bx") # bx = blue x
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.show()

print('it workin')
