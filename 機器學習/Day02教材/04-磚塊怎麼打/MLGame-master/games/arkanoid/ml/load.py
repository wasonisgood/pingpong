import pickle

# open a file, where you stored the pickled data
file = open('../games/arkanoid/log/2020-02-08_14-36-14.pickle', 'rb')
print(file)
# dump information to that file
data = pickle.load(file)

# close the file
file.close()

print('Showing the pickled data:')

cnt = 0
for item in data:
    print('The data ', cnt, ' is : ', item)
    cnt += 1