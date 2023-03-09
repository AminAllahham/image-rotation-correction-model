# Load image form disk and apply image processing
from PIL import Image

img = Image.open('test.jpg')

img = img.resize((224,224), Image.ANTIALIAS)

# create a copy & save the image
img.save('test_resized.jpg')