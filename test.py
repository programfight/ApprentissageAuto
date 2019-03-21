import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from skimage import data, exposure


image = Image.open('kebab.jpeg')
image = image.resize([250,250])
fd, h = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(h, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
