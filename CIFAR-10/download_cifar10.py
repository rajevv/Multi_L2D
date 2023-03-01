import os
import tarfile
import urllib.request

import torchvision

# Download CIFAR-10 ===
torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
torchvision.datasets.CIFAR10(root='../data', train=False, download=True)

# Delete the generated zip file
zip_file_path = "../data/cifar-10-python.tar.gz"
if os.path.exists(zip_file_path):
    os.remove(zip_file_path)
    print(f"Deleted the file {zip_file_path}.")
else:
    print(f"The file {zip_file_path} does not exist.")


# Download CIFAR-10C ===

url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
path_file = '../data/CIFAR-10-C' 
filename = 'CIFAR-10-C.tar'

os.chdir("../data/")
if not os.path.exists(path_file):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)

    print(f"Extracting {filename}...")
    with tarfile.open('../data/' + filename, 'r') as tar:
        tar.extractall()
else:
    print("CIFAR-10C already exists.")

# Delete the generated zip file
if os.path.exists(path_file + ".tar"):
    os.remove(path_file + ".tar")
    print(f"Deleted the file {zip_file_path}.")
else:
    print(f"The file {zip_file_path} does not exist.")
