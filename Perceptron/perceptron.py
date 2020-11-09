import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      rango de aprendizaje(entre 0.0 y 1.0)
    n_iter : int
        numero de ciclos      
    random_state : int  
      numero random usada como semilla generadora de los pesos random iniciales

    Attributes
    -----------
    w_ : 1d-array
    ES:
      peso despues del ajuste 
    ING:
      Weights after fitting.
    errors_ : list
    ING:
      Number of misclassifications (updates) in each epoch.
    ES:
      numero de clasificaciones erroneas en cada epoca (actualizacion)

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
            ajuste de datos de entrenamiento

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        ING:
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        
        ES:
        X: {array de forma}, forma = [n_muestra,n_caracteristicas]
          vectores de entrenamienro, donde n_muestra es el number de muestra y 
          n_caracteristicas es el numero de caracteristicas
          
        y : array-like, shape = [n_samples]
        ING:
          Target values.
        ES:
          valores objetivos

        Returns
        -------
        self : object

        """
        #numero semilla para el numero random
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])#X.shape[] regresa el numero de columnas de X-> vectores de entrenamiento,
        #donde el numero de columnas nos dice el numero de caracteristicas
        self.errors_ = []

        for _ in range(self.n_iter):#Cada una de las lineas de los datos de entrenamiento
            errors = 0 #el error inicial es 0 
            for xi, target in zip(X, y): # tomamos el n objeto de cada objeto iterable, vector entrenamiento ---- objetivo
                update = self.eta * (target - self.predict(xi)) #factor de aprendizaje(objetivo - valor predecido)
                self.w_[1:] += update * xi # multiplicamos el valor de actualizacion por el vector de caracteristicas y se lo sumamos a los vectores de peso de evaluacion
                self.w_[0] += update # es el valor de sesgo actualizadopor si se predijo mal se mejore
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ llega  esto [n_muestra,n_caracteristicas]"""
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]  #.dot hace una multiplicacion de matrices en 2D y el producto punto en 1D, siendo la que nos compete,
        # en este caso son dos arreglos con dos elementos y le vamos a sumar el primer elemento
        #multiplicacion del la fila Xn por w[1:]-> caracteristicas estandar, w[0]->sesgo
    def predict(self, X): #Recordar que la X que llega aqui es una parte del vector X principal es decir es [n_muestra,n_caracteristicas]
        """Return class label after unit step"""
        #Regresa clase
        return np.where(self.net_input(X) >= 0.0, 1, -1) # regresa 1 si la condicion es aceptada -1 si es negada
                                                         #la condicion esta en la funcion net_input()
   

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # definir un generador de marcadores y un mapa de colores
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # representar la superficie de decisión 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # representar muestras de clase
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
   
if (__name__ == "__main__"):
    df = pd.read_csv("/home/ray/Desktop/Machine_Learn/Perceptron/iris.data",header=None)
    
    #extraemos setosa y veriscolor y creamos vector de etiquetas
    Y = df.iloc[0:100, 4].values
    Y = np.where(Y == "Iris-setosa",-1,1)

    #extraemos caracteristicas longitud de sepalo y longitud de petalo                                                   
    X = df.iloc[0:100, [0,2]].values  
        
    ppn = Perceptron(eta=0.1,n_iter=10)
    ppn.fit(X,Y)
    plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
    plt.xlabel("Iteracion")
    plt.ylabel('Errores por Iteracion')
    plt.show() 
    
    #Representamos los datos
    plt.scatter(X[:50, 0], X[:50 ,1], color='red',marker='o',label='setosa')   
    plt.scatter(X[50:100, 0], X[50:100 ,1], color='blue',marker='x',label='veriscolor')              
    plt.xlabel('sepal lenght [cm]')               
    plt.ylabel('petal lenght [cm]')    
    plt.legend(loc ='upper left')  
    plt.show()   
    
    plot_decision_regions(X,Y,classifier=ppn)
    plt.xlabel('sepal lenght [cm]')               
    plt.ylabel('petal lenght [cm]')    
    plt.legend(loc ='upper left')  
    plt.show()        
    
    


                    
                    
                                                    