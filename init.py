import random
from mnist import MNIST 

data_path = r'D:\Datasets\mnist'

mndata =  MNIST(data_path)

images, labels = mndata.load_training()

index = random.randrange(0, len(images))  
print(mndata.display(images[index]))