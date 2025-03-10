import numpy as np
import matplotlib.pyplot as plt

black_square = np.zeros((50, 50))
white_square = np.ones((50, 50)) * 255

top_row = np.hstack((black_square, white_square))
bottom_row = np.hstack((white_square, black_square))
checkerboard = np.vstack((top_row, bottom_row))

plt.imshow(checkerboard, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title("Crno-bijeli uzorak")
plt.show()
