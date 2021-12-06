
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score,f1_score,precision_score,recall_score


def validation(rede, X_test_std,y_test, p = True):
  y_pred = rede.predict(X_test_std)
  cmatrix = confusion_matrix(y_test,y_pred)
  
  #6.2 Acurácia
  acc = balanced_accuracy_score(y_test,y_pred,adjusted=False)
  

  #6.4 Precisão
  pre = precision_score(y_test,y_pred,average='weighted')
  

  #6.5 Revocação 
  rev = recall_score(y_test,y_pred,average='weighted')
  

  #6.3 F-Score
  f1 = f1_score(y_test,y_pred,average='weighted')

  if (p):
    print("Matriz de confusão:\n")
    print(cmatrix)
    print("\nAcurácia: ",acc)
    print("\nPrecisão: ",pre)
    print("\nRevocação: ",rev)
    print("\nF-Score: ",f1)
  return cmatrix, acc, pre, rev, f1

path = '/home/elloa/rna2021.1/covtype.csv'
df = pd.read_csv(path)

#2 e 3
X = df[df.columns[0:10]]
y = df[df.columns[-1]]

#4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,train_size=0.7,random_state=42, shuffle=True)

#5
X_train_std = (X_train - np.mean(X_train))/np.std(X_train)
X_test_std = (X_test - np.mean(X_train))/np.std(X_train)

#5
rede = MLPClassifier(hidden_layer_sizes=(10),activation='relu',max_iter=300)
rede.fit(X_train_std,y_train)

validation(rede,X_test_std, y_test)

resultados = []
for x in range(15):
  #7 e 7.1
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,train_size= 0.7, shuffle=True)
  
  ## Padronização
  X_train_std = (X_train - np.mean(X_train))/np.std(X_train)
  X_test_std = (X_test - np.mean(X_train))/np.std(X_train)
  
  ### Modelo
  rede1 = MLPClassifier(hidden_layer_sizes=(10),activation='relu',max_iter=300,verbose=False)
  rede1.fit(X_train_std,y_train)

  resultados.append(validation(rede1, X_test, y_test, p = False))

for resultado in resultados:
  print(resultado)

soma_acc = 0
soma_f1score = 0
for resultado in resultados:
  soma_acc+= resultado[1]
  soma_f1score+= resultado[4]
media_acc = soma_acc/len(resultados)
media_f1 = soma_f1score/len(resultados)
print(f"medias:\nAcurácia: {media_acc}\nf1 score:{media_f1}")

"""#### Desvio padrão"""

somatorio_acc = 0
somatorio_f1 = 0
for resultado in resultados:
  somatorio_acc += (resultado[1] - media_acc)**(2)
  somatorio_f1 += (resultado[4] - media_f1)**(2)

dp_acc = (somatorio_acc/len(resultado))**(1/2)
dp_f1 = (somatorio_f1/len(resultado))**(1/2)
print(f"desvio padrao:\nAcurácia: {dp_acc}\nf1 score:{dp_f1}")

resultados = []
for x in range(15):
  #7 e 7.1
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,train_size= 0.7, shuffle=True)
  
  ## Padronização
  X_train_std = (X_train - np.mean(X_train))/np.std(X_train)
  X_test_std = (X_test - np.mean(X_train))/np.std(X_train)
  
  ### Modelo
  rede1 = MLPClassifier(hidden_layer_sizes=(10),activation='relu',max_iter=300,verbose=False, solver = 'sgd')
  rede1.fit(X_train_std,y_train)

  resultados.append(validation(rede1, X_test, y_test, p = False))

for resultado in resultados:
  print(resultado)

soma_acc = 0
soma_f1score = 0
for resultado in resultados:
  soma_acc+= resultado[1]
  soma_f1score+= resultado[4]
media_acc = soma_acc/len(resultados)
media_f1 = soma_f1score/len(resultados)
print(f"medias:\nAcurácia: {media_acc}\nf1 score:{media_f1}")



somatorio_acc = 0
somatorio_f1 = 0

for resultado in resultados:
  somatorio_acc += (resultado[1] - media_acc)**(2)
  somatorio_f1 += (resultado[4] - media_f1)**(2)

dp_acc = (somatorio_acc/len(resultado))**(1/2)
dp_f1 = (somatorio_f1/len(resultado))**(1/2)
print(f"desvio padrao:\nAcurácia: {dp_acc}\nf1 score:{dp_f1}")

#1
#Uma ou duas camadas ocultas
#solver Adam ou SGD
# épocas: 100, 150 ou 200
redes = [[] for i in range(10)]
redes[0] = MLPClassifier(hidden_layer_sizes=(10),activation='tanh',max_iter=100,solver="adam")
redes[1] = MLPClassifier(hidden_layer_sizes=(10),activation='relu',max_iter=150,solver="sgd")
redes[2] = MLPClassifier(hidden_layer_sizes=(10),activation='logistic',max_iter=200,solver="adam")
redes[3] = MLPClassifier(hidden_layer_sizes=(10,15),activation='identity',max_iter=100,solver="adam")
redes[4] = MLPClassifier(hidden_layer_sizes=(10,15),activation='relu',max_iter=150,solver="adam")
redes[5] = MLPClassifier(hidden_layer_sizes=(10,15),activation='tanh',max_iter=200,solver="adam")
redes[6] = MLPClassifier(hidden_layer_sizes=(20),activation='relu',max_iter=100,solver="adam")
redes[7] = MLPClassifier(hidden_layer_sizes=(20),activation='logistic',max_iter=150,solver="adam")
redes[8] = MLPClassifier(hidden_layer_sizes=(20),activation='identity',max_iter=200,solver="adam")
redes[9] = MLPClassifier(hidden_layer_sizes=(30,40),activation='logistic',max_iter=200,solver="adam")

#2 Avaliar cada rede 15 vezes

#métricas
matriz_c = [[] for i in range(len(redes))]
acc = [[] for i in range(len(redes))]
pre = [[] for i in range(len(redes))]
rev = [[] for i in range(len(redes))]
f1 = [[] for i in range(len(redes))]

for i in range(15):
  #Inicialização
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,train_size=0.7,shuffle=True)

  #Normalização
  X_train_std = (X_train - np.mean(X_train))/np.std(X_train)
  X_test_std = (X_test - np.mean(X_train))/np.std(X_train)

  for j in range(len(redes)):
    redes[j].fit(X_train_std,y_train)
    a,b,c,d,e = validation(redes[j], X_test_std,y_test, p = False)

    
    matriz_c[j].append(a)
    acc[j].append(b)
    pre[j].append(c)
    rev[j].append(d)
    f1[j].append(e)

#3
#3.1 Top-3 melhores redes no tocante à F-Score e Acurácia

#Usando a média para descobrir a melhor rede
acc_m =[[] for i in range(len(acc))]
f1_m =[[] for i in range(len(f1))]
resultado = [[] for i in range(len(f1))]
indices = [i for i in range(len(f1))]
#Média das Acurácias
for i in range(len(acc)):
  aux = 0
  for j in range(len(acc[i])):
    aux += acc[i][j]

  acc_m[i] = (aux/(j+1))

#Média dos F_Scores
for i in range(len(f1)):
  aux = 0
  for j in range(len(f1[i])):
    aux += f1[i][j]

  f1_m[i] = (aux/(j+1))


# Soma dos valores
#As redes com maior soma de acurácia e f-score serão selecionadas
for i in range(len(resultado)):
  resultado[i] = acc_m[i] + f1_m[i]

resultado1 = sorted(zip(resultado,indices), reverse=True)[:3]
print("Melhores redes:\n")
for i in resultado1:

  print("Rede de índice: ",i[1])
  print("Acurácia Média: ",acc_m[i[1]])
  print("F-Score Médio: ",f1_m[i[1]])
  print('')

#3.2 Repetição em que houve o melhor desempenho de cada uma dessas redes: ilustre tp, tf, fp e fn

#Melhor desempenho = maior valor de f-score + acurácia
desempenho = [[] for i in range(len(resultado1))]
for i in range(len(resultado1)):
  indice = resultado1[i][1]
  for j in range(len(acc[indice])):
    desempenho[i].append([acc[indice][j]+f1[indice][j],j])

desempenho1 = []

for i in range(len(desempenho)):
  desempenho1.append(max(desempenho[i]))

for i in range(len(resultado1)):
  print("\nO melhor desempenho da rede",resultado1[i][1]," foi na repetição:",desempenho1[i][1]+1)
  print("\n Matriz de Confusão: ")
  m = matriz_c[resultado1[i][1]][desempenho1[i][1]]
  print(m)

#3
#Nh1 = 4
redes1 = [[] for i in range(30)]
redes1[0] = MLPClassifier(hidden_layer_sizes=(4),activation='tanh',max_iter=100,solver="adam")
redes1[1] = MLPClassifier(hidden_layer_sizes=(4),activation='relu',max_iter=150,solver="sgd")
redes1[2] = MLPClassifier(hidden_layer_sizes=(4),activation='logistic',max_iter=200,solver="adam")
redes1[3] = MLPClassifier(hidden_layer_sizes=(4),activation='identity',max_iter=100,solver="adam")
redes1[4] = MLPClassifier(hidden_layer_sizes=(4),activation='relu',max_iter=200,solver="adam")
redes1[5] = MLPClassifier(hidden_layer_sizes=(4,4),activation='tanh',max_iter=100,solver="adam")
redes1[6] = MLPClassifier(hidden_layer_sizes=(4,4),activation='relu',max_iter=100,solver="adam")
redes1[7] = MLPClassifier(hidden_layer_sizes=(4,4),activation='logistic',max_iter=150,solver="adam")
redes1[8] = MLPClassifier(hidden_layer_sizes=(4,4),activation='identity',max_iter=200,solver="adam")
redes1[9] = MLPClassifier(hidden_layer_sizes=(4,4),activation='tanh',max_iter=200,solver="adam")

#Nh1 = 16
redes1[10] = MLPClassifier(hidden_layer_sizes=(16),activation='tanh',max_iter=100,solver="adam")
redes1[11] = MLPClassifier(hidden_layer_sizes=(16),activation='relu',max_iter=150,solver="sgd")
redes1[12] = MLPClassifier(hidden_layer_sizes=(16),activation='logistic',max_iter=200,solver="adam")
redes1[13] = MLPClassifier(hidden_layer_sizes=(16),activation='identity',max_iter=100,solver="adam")
redes1[14] = MLPClassifier(hidden_layer_sizes=(16),activation='relu',max_iter=200,solver="adam")
redes1[15] = MLPClassifier(hidden_layer_sizes=(16,16),activation='tanh',max_iter=100,solver="adam")
redes1[16] = MLPClassifier(hidden_layer_sizes=(16,16),activation='relu',max_iter=100,solver="adam")
redes1[17] = MLPClassifier(hidden_layer_sizes=(16,16),activation='logistic',max_iter=150,solver="adam")
redes1[18] = MLPClassifier(hidden_layer_sizes=(16,16),activation='identity',max_iter=200,solver="adam")
redes1[19] = MLPClassifier(hidden_layer_sizes=(16,16),activation='tanh',max_iter=200,solver="adam")

#Nh1 = 25
redes1[20] = MLPClassifier(hidden_layer_sizes=(25),activation='tanh',max_iter=100,solver="adam")
redes1[21] = MLPClassifier(hidden_layer_sizes=(25),activation='relu',max_iter=150,solver="sgd")
redes1[22] = MLPClassifier(hidden_layer_sizes=(25),activation='logistic',max_iter=200,solver="adam")
redes1[23] = MLPClassifier(hidden_layer_sizes=(25),activation='identity',max_iter=100,solver="adam")
redes1[24] = MLPClassifier(hidden_layer_sizes=(25),activation='relu',max_iter=200,solver="adam")
redes1[25] = MLPClassifier(hidden_layer_sizes=(25,25),activation='tanh',max_iter=100,solver="adam")
redes1[26] = MLPClassifier(hidden_layer_sizes=(25,25),activation='relu',max_iter=100,solver="adam")
redes1[27] = MLPClassifier(hidden_layer_sizes=(25,25),activation='logistic',max_iter=150,solver="adam")
redes1[28] = MLPClassifier(hidden_layer_sizes=(25,25),activation='identity',max_iter=200,solver="adam")
redes1[29] = MLPClassifier(hidden_layer_sizes=(25,25),activation='tanh',max_iter=200,solver="adam")

#3.1
#2 Avaliar cada rede 15 vezes

#métricas
matriz_c = [[] for i in range(len(redes1))]
acc = [[] for i in range(len(redes1))]
pre = [[] for i in range(len(redes1))]
rev = [[] for i in range(len(redes1))]
f1 = [[] for i in range(len(redes1))]

for i in range(15):
  #Inicialização
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,train_size=0.7,shuffle=True)

  #Normalização
  X_train_std = (X_train - np.mean(X_train))/np.std(X_train)
  X_test_std = (X_test - np.mean(X_train))/np.std(X_train)

  for j in range(len(redes1)):
    redes1[j].fit(X_train_std,y_train)
    a,b,c,d,e = validation(redes1[j], X_test_std,y_test, p = False)

    
    matriz_c[j].append(a)
    acc[j].append(b)
    pre[j].append(c)
    rev[j].append(d)
    f1[j].append(e)

#-------------------------------------------------------------------------------
#Usando a média para descobrir a melhor rede
acc_m =[[] for i in range(len(acc))]
f1_m =[[] for i in range(len(f1))]
resultado = [[] for i in range(len(f1))]
indices = [i for i in range(len(f1))]
#Média das Acurácias
for i in range(len(acc)):
  aux = 0
  for j in range(len(acc[i])):
    aux += acc[i][j]
  acc_m[i] = (aux/(j+1))

#Média dos F_Scores
for i in range(len(f1)):
  aux = 0
  for j in range(len(f1[i])):
    aux += f1[i][j]

  f1_m[i] = (aux/(j+1))


# Soma dos valores
#As redes com maior soma de acurácia e f-score serão selecionadas
for i in range(len(resultado)):
  resultado[i] = acc_m[i] + f1_m[i]

resultado2 = sorted(zip(resultado,indices), reverse=True)[:3]

print("Melhores redes:\n")
for i in resultado2:
  
  print("Rede de índice: ",i[1])
  print("Acurácia Média: ",acc_m[i[1]])
  print("F-Score Médio: ",f1_m[i[1]])
  print('')

#1
redes2 = []
for i in range(len(resultado1)):  
  redes2.append(redes[resultado1[i][1]])

for i in range(len(resultado2)):
  redes2.append(redes1[resultado2[i][1]])

#2.1
for i in range(len(redes2)):
  redes2[i].max_iter=200

#2
X = df[df.columns[0:54]]
y = df[df.columns[-1]]

#métricas
matriz_c = [[] for i in range(len(redes2))]
acc = [[] for i in range(len(redes2))]
pre = [[] for i in range(len(redes2))]
rev = [[] for i in range(len(redes2))]
f1 = [[] for i in range(len(redes2))]

for i in range(15):
  #Inicialização
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,train_size=0.7,shuffle=True)

  #Normalização
  X_train_std = (X_train - np.mean(X_train))/np.std(X_train)
  X_test_std = (X_test - np.mean(X_train))/np.std(X_train)

  for j in range(len(redes2)):
    redes1[j].fit(X_train_std,y_train)
    a,b,c,d,e = validation(redes1[j], X_test_std,y_test, p = False)

    
    matriz_c[j].append(a)
    acc[j].append(b)
    pre[j].append(c)
    rev[j].append(d)
    f1[j].append(e)

#-------------------------------------------------------------------------------
acc_m =[[] for i in range(len(acc))]
f1_m =[[] for i in range(len(f1))]
resultado = [[] for i in range(len(f1))]
indices = [i for i in range(len(f1))]
dp_acc = [[] for i in range(len(acc))]
dp_f1 = [[] for i in range(len(f1))]
#Média das Acurácias
for i in range(len(acc)):
  aux = 0
  for j in range(len(acc[i])):
    aux += acc[i][j]
  acc_m[i] = (aux/(j+1))

#Média dos F_Scores
for i in range(len(f1)):
  aux = 0
  for j in range(len(f1[i])):
    aux += f1[i][j]
  
  f1_m[i] = (aux/(j+1))

#DP das Acurácias

for i in range(len(acc)):
  aux = 0
  for j in range(len(acc[i])):
    aux += (acc[i][j] - acc_m[i])**2

  dp_acc[i] = (aux/(j+1))**(1/2)


#DP dos F-Scores
for i in range(len(f1)):
  aux = 0
  for j in range(len(f1[i])):
    aux += (f1[i][j] - f1_m[i])**2

  dp_f1[i] = (aux/(j+1))**(1/2)

#2.2
for i in range(len(redes2)):
  print("Rede de índice: ",i)
  print("Acurácia média +- dp:")
  print(str(acc_m[i])+" +- "+str(dp_acc[i]))
  print('')
  print("F-Score médio +- dp:")
  print(str(f1_m[i])+" +- "+str(dp_f1[i]))
  print('')
  print('-------------------------------')

print("Prints do F1 Score:\n")
for item in f1:
  print(item)