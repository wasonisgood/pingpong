import pickle
from os import path

import numpy as np
# Fetch data
filename = path.join(path.dirname(__file__), '2020-02-08_14-36-14.pickle')
log = pickle.load((open(filename, 'rb')))

Frames = []
Balls = []
Commands = []
PlatformPos = []
sceneInfo = log[0]
for sceneInfo in log:
    Frames.append(sceneInfo.frame)
    Balls.append([sceneInfo.ball[0],sceneInfo.ball[1]])
    # Commands.append(sceneInfo.command)
    PlatformPos.append(sceneInfo.platform)
    if sceneInfo.command == 'RIGHT':
        Commands.append('RIGHT')
    elif sceneInfo.command == 'LEFT':
        Commands.append('LEFT')
    else:
        Commands.append('NONE')
# print(Balls)
def compute_x_end(ball, ball_last):
    direction_x = ball[0] - ball_last[0]
    direction_y = ball[1] - ball_last[1]
    ball_x_end = 0
    # y = mx + c
    if direction_y>0:
        m = direction_y / direction_x
        c = ball[1] - m*ball[0]
        ball_x_end = (400 - c )/m
    else:
        ball_x_end = 100
    while ball_x_end < 0 or ball_x_end > 200:
        if ball_x_end<0:
            ball_x_end = -ball_x_end
        elif ball_x_end>200:
            ball_x_end = 400-ball_x_end
    # print(ball_x_end)
    return ball_x_end

def getX_pos(balls,balls_last):
    x_pos = []
    for i in range(0, len(balls)):
        x_pos.append( compute_x_end(balls[i],balls_last[i]))
    return x_pos
# Prepossessing
PlatX = np.array(PlatformPos)[:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]

Balls = np.array(Balls)
Balls_next = np.array(Balls[1:])

vectors = Balls_next - Balls[:-1]
vectors = vectors[:,0]/vectors[:,1]
vectors = vectors.reshape(len(vectors),1)
print('vectors = ',vectors)

instruct = getX_pos(Balls_next,Balls)


Ballarray = np.array(Balls[1:])
x = np.hstack((Ballarray,vectors))
# print(x.shape)
y = np.array(instruct)
print('y=',y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y)

# training
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
# svm = SVR(C=1000, epsilon=0.01)
nn = MLPRegressor(activation='relu',  hidden_layer_sizes=[50, 50])
# model = svm.fit(x,y)
model = nn.fit(x,y)
# print(x_test)
y_predict = model.predict(x_test)
print(y_predict)
acc=model.score(x_test,y_test)
# output
print(acc)

filename =  "model.sav"
modelfile = path.join(path.dirname(__file__),'../games/arkanoid/ml',filename)
pickle.dump(model, open(modelfile, 'wb'))