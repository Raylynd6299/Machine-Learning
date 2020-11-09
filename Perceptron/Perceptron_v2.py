import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    def __init__(self,taza_crecimineto = 0.1,num_iteraciones = 50, semilla_random = 1):
        self.taz_crec = taza_crecimineto
        self.n_iter = num_iteraciones
        self.sem_rand = semilla_random
    def entrenamiento(self,caracteristicas, objetivos):
        random_gen = np.random.RandomState(self.sem_rand)
        self.pesos_ = random_gen.normal(loc= 0.0, scale = 0.01,size = 1 + caracteristicas.shape[1])
        self.errores_ = []
        
        for _ in range(self.n_iter):
            error = 0
            for carac_i, objetivo in zip(caracteristicas,objetivos):
                ajuste = self.taz_crec * (objetivo - self.prediccion(carac_i))
                self.pesos_[1:] += ajuste * carac_i
                self.pesos_[0] += ajuste
                error += int(ajuste != 0.0)
            self.errores_.append(error)
        return self
    def ingreso_de_red(self,caracteristicas):
        return np.dot(caracteristicas,self.pesos_[1:]) + self.pesos_[0]
    
    def prediccion(self,caracteristicas):
        return np.where(self.ingreso_de_red(caracteristicas)>=0.0,1,-1)
    
def plot_regiones_de_decicion(caracteristicas,objetivos,clasificador,resolucion=0.2):
    marcadores = ('s', 'x', 'o', '^', 'v')
    colores = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    mapa_colores = ListedColormap(colores[:len(np.unique(objetivos))])
    
    carac1_min,carac1_max = caracteristicas[:,0].min(),caracteristicas[:,0].max()
    carac2_min,carac2_max = caracteristicas[:,1].min(),caracteristicas[:,1].max()
    
    
    mat_carac1,mat_carac2 = np.meshgrid(np.arange(carac1_min,carac1_max,resolucion),
                                        np.arange(carac2_min,carac2_max,resolucion))
    
    z = clasificador.prediccion(np.array([mat_carac1.ravel(),mat_carac2.ravel()]).T )
    z = z.reshape(mat_carac1.shape)
    
    plt.contour(mat_carac1,mat_carac2,z,alpha=0.3,cmap=mapa_colores)
    plt.xlim(carac1_min,carac1_max)
    plt.ylim(carac2_min,carac2_max)

    for idx, cl in enumerate(np.unique(objetivos)):
        plt.scatter(x=caracteristicas[objetivos== cl, 0], 
                    y=caracteristicas[objetivos == cl, 1],
                    alpha=0.8, 
                    c=colores[idx],
                    marker=marcadores[idx], 
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
    
    plot_regiones_de_decicion(X,Y,classifier=ppn)
    plt.xlabel('sepal lenght [cm]')               
    plt.ylabel('petal lenght [cm]')    
    plt.legend(loc ='upper left')  
    plt.show()        
    
    