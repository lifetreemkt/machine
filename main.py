import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import metrics


data =pd.read_excel('./base/dataset.xlsx')
#verificando tipo de dados >>> print (data.dtypes)

# criando dicionario passando coluna portatil como chave

alterando =  {'Portatil': {'Smartphone':1, 'Tablet':2}}

#inserir alteações no excel
data.replace(alterando, inplace=True)

y = data.Portatil
X = data.drop(columns=['Portatil'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#inicio do aprendizado
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

#iniciar predição com base no aprendizado
y_pred = clf.predict(X_test)
y_true = y_test

print(f'Computador = {y_pred}, Gabarito = {y_true.values}')

print(str(metrics.precision_score(y_pred,y_true)))
