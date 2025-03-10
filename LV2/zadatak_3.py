import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("road.jpg")

# a)
brighter = np.clip(img.astype(np.float32) + 50, 0, 255).astype(np.uint8)

# b)
height, width, _ = img.shape
quarter = img[:, width//4:width//2, :]

# c)
rotated = np.rot90(img, k=-1)

# d)
mirrored = np.fliplr(img)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(brighter)
axs[0, 0].set_title("Posvijetljena slika")
axs[0, 0].axis("off")

axs[0, 1].imshow(quarter)
axs[0, 1].set_title("Druga četvrtina slike")
axs[0, 1].axis("off")

axs[1, 0].imshow(rotated)
axs[1, 0].set_title("Rotirana za 90°")
axs[1, 0].axis("off")

axs[1, 1].imshow(mirrored)
axs[1, 1].set_title("Zrcaljena slika")
axs[1, 1].axis("off")

plt.show()
