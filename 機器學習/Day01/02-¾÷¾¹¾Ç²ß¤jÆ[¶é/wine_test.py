#%%
"""練習Lab02-1:使用DecisionTree 預測酒的種類，並畫出樹狀圖。
複習重點1.了解如何使用不同的資料集
複習重點2.了解資料科學領域的步驟
複習重點3.嘗試了解不同分類器不同參數的用途。
"""
from sklearn.datasets import load_wine

wine = load_wine()
from sklearn.model_selection import train_test_split
data = wine.data
target = wine.target
enddata=data[:,0:4]
x_train,x_test,y_train,y_test=train_test_split(enddata,target)
print("選",len(x_train),"訓練")
print("選",len(x_test),"測試")
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
wine_clf=knn.fit(x_train,y_train)
print(wine_clf.__class__)
y_predict = wine_clf.predict(x_test)
print("原始結果 original result:")
print(y_test)
print("預測結果 predicted result:")
print(y_predict)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_predict)
print("Accuracy(正確率) ={:8.3f}%".format(accuracy*100))




# %%
