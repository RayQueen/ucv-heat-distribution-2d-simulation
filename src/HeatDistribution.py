from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

# Funcion para generar los índices
def node_index(row, col):
    return row * n + col

# Funcion para transformar el vector solucion a una matriz con las fronteras incluidas
def reshape_vector(solution, n, t_top, t_bottom, t_left, t_right):
    field = np.zeros((n + 2, n + 2), dtype=float)   # Crea una matriz para la solucion con espacio para las fronteras
    field[0, :] = t_bottom                          # Asigna la temperatura de la frontera inferior
    field[-1, :] = t_top                            # Asigna la temperatura de la frontera superior
    field[:, 0] = t_left                            # Asigna la temperatura de la frontera izquierda
    field[:, -1] = t_right                          # Asigna la temperatura de la frontera derecha

    interior = solution.reshape((n, n))             # Reshape del vector solucion a una matriz 2D de tamaño n x n para los nodos internos
    field[1:-1, 1:-1] = interior                    # Asigna los valores de la solucion a la parte interior de la matriz, dejando las fronteras intactas
    return field

# 0) Parametros del sistema
n = 20          # Número de nodos internos por lado
t_top = 100.0   # Temperatura en la frontera superior
t_bottom = 0.0  # Temperatura en la frontera inferior
t_left = 50.0   # Temperatura en la frontera izquierda
t_right = 50.0  # Temperatura en la frontera derecha

# Construye el sistema lineal para la ecuacion de Poisson en una placa cuadrada.
h = 1.0 / (n + 1)                       # Distancia entre nodos internos
size = n * n                            # Numero total de nodos internos
A = np.zeros((size, size), dtype=float) # Matriz de coeficientes
b = np.zeros(size, dtype=float)         # Vector de términos independientes

# 1) Construccion de la matriz A y el vector b
for row in range(n):
    for col in range(n):
        k = node_index(row, col) # Indice del nodo actual en la matriz A

        A[k, k] = 4.0 # Coeficiente del nodo actual dado por la ecuacion

        # Si el nodo es interno, sus vecinos contribuyen a la matriz A con coeficientes -1
        # Si su vecino es una frontera, contribuye al vector b con la temperatura de la frontera correspondiente

        if row > 0:
            A[k, node_index(row - 1, col)] = -1.0
        else:
            b[k] += t_bottom

        if row < n - 1:
            A[k, node_index(row + 1, col)] = -1.0
        else:
            b[k] += t_top

        if col > 0:
            A[k, node_index(row, col - 1)] = -1.0
        else:
            b[k] += t_left

        if col < n - 1:
            A[k, node_index(row, col + 1)] = -1.0
        else:
            b[k] += t_right

# 2) Fuente de calor interna en el centro de la placa

source_strength = 50000.0 # Intensidad de la fuente de calor

# Si n es par, el centro se encuentra entre cuatro nodos internos
if n % 2 == 0:
    center_indices = [
        node_index(n // 2 - 1, n // 2 - 1),
        node_index(n // 2 - 1, n // 2),
        node_index(n // 2, n // 2 - 1),
        node_index(n // 2, n // 2)
    ]
else: # Si n es impar, el centro coincide con un nodo interno
    center_index = node_index(n // 2, n // 2)

b_with_source = b.copy() # Copia del vector b para agregar la fuente de calor
if n % 2 == 0: # Si n es par, distribuye la fuente de calor entre los cuatro nodos centrales
    for idx in center_indices:
        b_with_source[idx] += source_strength * h**2 / 4
else: # Si n es impar, asigna toda la fuente de calor al nodo central
    b_with_source[center_index] += source_strength * h**2

# 3) Resolucion del sistema lineal Au = b para encontrar la distribucion de temperatura en los nodos internos u

# Convierte la matriz A en una matriz dispersa
A_sparse = scipy.sparse.csr_matrix(A) # Formato CSR para eficiencia en la resolucion de sistemas lineales dispersos

# Resuelve el sistema lineal para obtener la distribucion de temperatura en los nodos internos sin y con la fuente de calor
u_without_source = scipy.sparse.linalg.spsolve(A_sparse, b)             # Sin fuente de calor
u_with_source = scipy.sparse.linalg.spsolve(A_sparse, b_with_source)    # Con fuente de calor

# Convertir el resultado (vector u) a una matriz 2D para visualizacion
reshaped_with_source = reshape_vector(u_with_source, n, t_top, t_bottom, t_left, t_right)       # Reshape del vector u a una matriz 2D de tamaño n x n
reshaped_without_source = reshape_vector(u_without_source, n, t_top, t_bottom, t_left, t_right) # Reshape del vector u a una matriz 2D de tamaño n x n

# 4) Visualizacion de la distribucion de temperatura

# Directorio para guardar las imagenes de salida
output_path = Path("img") 
output_path.mkdir(parents=True, exist_ok=True)

# 1. Matriz del sistema: patron de bandas.
plt.figure(figsize=(6.5, 6.5))
plt.spy(A_sparse, markersize=1)
plt.title("Patron de Bandas de la Matriz A")
plt.xlabel("Columnas")
plt.ylabel("Filas")
plt.tight_layout()
plt.savefig(output_path / "01_patron_bandas_matriz_A.png", dpi=200)

# 2. Solucion numerica: mapa de calor con curvas de nivel.

# Crear un grid para la visualizacion
x = np.linspace(0.0, 1.0, n + 2)
y = np.linspace(0.0, 1.0, n + 2)
xx, yy = np.meshgrid(x, y)

plt.figure(figsize=(8, 6.5))
heatmap = plt.contourf(xx, yy, reshaped_with_source, levels=30, cmap="inferno")
contours = plt.contour(xx, yy, reshaped_with_source, levels=12, colors="white", linewidths=0.7, alpha=0.75)
plt.clabel(contours, inline=True, fontsize=8, fmt="%1.0f")
plt.colorbar(heatmap, label="Temperatura (°C)")
plt.title("Distribucion de Temperatura con Fuente de Calor")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(output_path / "02_heatmap_con_isotermas.png", dpi=200)

# 3. Analisis fisico: comparacion sin y con fuente.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
plot0 = axes[0].contourf(xx, yy, reshaped_without_source, levels=30, cmap="inferno")
axes[0].contour(xx, yy, reshaped_without_source, levels=12, colors="white", linewidths=0.6, alpha=0.65)
axes[0].set_title("Sin fuente de calor")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

plot1 = axes[1].contourf(xx, yy, reshaped_with_source, levels=30, cmap="inferno")
axes[1].contour(xx, yy, reshaped_with_source, levels=12, colors="white", linewidths=0.6, alpha=0.65)
axes[1].set_title("Con fuente de calor")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

fig.colorbar(plot1, ax=axes, shrink=0.95, label="Temperatura (°C)")
fig.savefig(output_path / "03_comparacion_con_y_sin_fuente.png", dpi=200)

# 4. Perfil transversal por el centro.
center_row = int(np.argmin(np.abs(y - 0.5)))
plt.figure(figsize=(8, 5.2))
plt.plot(x, reshaped_without_source[center_row, :], label="Sin fuente", linewidth=2)
plt.plot(x, reshaped_with_source[center_row, :], label="Con fuente", linewidth=2)
plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.8)
plt.title("Perfil transversal u(x, 0.5)")
plt.xlabel("x")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_path / "04_perfil_transversal_central.png", dpi=200)

plt.close("all")