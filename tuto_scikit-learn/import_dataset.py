from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

print("import of the datasets done")

print("\ndigits data (all images): \n",digits.data) # print images arrays
print("\nwanted outputs:\n",digits.target) # the wanted values for each image

print("\nfirst image:\n",digits.images[0]) # print first image

