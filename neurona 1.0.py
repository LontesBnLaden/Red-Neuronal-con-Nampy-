import numpy as np

# Función sigmoid y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

# Datos de ejemplo: kilos comprados de [manzana, banana, naranja]
# Cada fila es una compra diferente
entradas = np.array([
    [1, 2, 1],    # cliente 1
    [2, 1, 0],    # cliente 2
    [3, 0, 2],    # cliente 3
    [0, 1, 3],    # cliente 4
    [4, 2, 1],    # cliente 5
])

# Gasto total real en dólares (lo que se pagó en cada compra)
salidas = np.array([[10], [8], [12], [9], [15]])

# Inicializar pesos aleatorios
np.random.seed(1)
pesos_entrada_oculta = 2 * np.random.random((3, 4)) - 1  # 3 entradas a 4 neuronas
pesos_oculta_salida = 2 * np.random.random((4, 1)) - 1   # 4 neuronas a 1 salida

# Entrenamiento
for _ in range(10000):
    # Forward pass
    capa_entrada = entradas
    capa_oculta = sigmoid(np.dot(capa_entrada, pesos_entrada_oculta))
    salida_predicha = sigmoid(np.dot(capa_oculta, pesos_oculta_salida))
    
    # Error
    error = salidas - salida_predicha
    
    # Backpropagation
    delta_salida = error * sigmoid_derivada(salida_predicha)
    delta_oculta = delta_salida.dot(pesos_oculta_salida.T) * sigmoid_derivada(capa_oculta)
    
    # Ajuste de p
