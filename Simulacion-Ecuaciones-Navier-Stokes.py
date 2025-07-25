"""
Autores:
Jose Miguel Fuertes Benavides - 202224623
Heidy Lizbeth Gelpud Acosta - 2242550
Esteban Samuel Cordoba Narvaez - 202370976

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline


def initialize_grid(nx, ny):
    """Inicializa la malla de velocidades."""
    v = np.zeros((nx, ny), float)
    return v

def apply_boundary_conditions(v):
    """Aplica las condiciones de frontera."""
    v[0, :] = 1  # Lado izquierdo = 1
    v[-1, :] = 0  # Lado derecho = 0
    v[:, 0] = 0  # Lado superior = 0
    v[:, -1] = 0  # Lado inferior = 0
    return v

def compute_residual(v, nx, ny):
    """Calcula el residuo F(v) con un caso especial para i == 1."""
    F = np.zeros_like(v)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
                if i == 1 and j == 1:#Esquina Superior Izquierda
                    F[i, j] = v[1,1] - 0.25 * ((0.5 * v[2, 1]) + 1.5 + (0.975 * v[1,2]))
                elif i == nx - 2 and j == 1:#Esquina Superior Derecha
                    F[i, j] = v[i, j] - 0.25 * ((1.5 * v[nx - 3, 1]) + (0.975 * v[nx - 2, 2]))
                elif i == 1 and j == ny - 2:#Esquina Inferior Izquierda
                    F[i, j] = v[i, j] - 0.25 * ((0.5 * v[2, ny - 2]) + 1.5 + (1.025 * v[1, ny - 3]))
                elif i == nx - 2 and j == ny - 2:#Esquina Inferior Derecha
                    F[i, j] = v[i, j] - 0.25 * ((1.5 * v[nx - 3, ny - 2]) + (1.025 * v[nx - 2, ny - 3]))
                elif j == 1:#Arriba
                    F[i, j] = v[i, j] - 0.25 * ((0.5 * v[i+1, 1]) + (1.5 * v[i-1, 1]) + (0.975 * v[i, 2]))
                elif i == 1:#Izquierda
                    F[i, j] = v[i, j] - 0.25 * ((0.5 * v[2, j]) + 1.5 + (0.975 * v[1, j+1]) + (1.025 * v[1, j-1]))
                elif i == nx - 2:#Derecha
                    F[i, j] = v[i, j] - 0.25 * ((1.5 * v[nx - 2, j]) + (0.975 * v[nx - 2, j+1]) + (1.025* v[nx - 2, j-1]))
                elif j == ny - 2: #Abajp
                    F[i, j] = v[i, j] - 0.25 * ((0.5 * v[i+1, ny - 2]) + (1.5 * v[i-1, ny - 2]) + (1.025 * v[i, ny - 3]))
                else: #Centro
                    F[i, j] = v[i, j] - 0.25 * ((0.5 * v[i+1, j]) + (1.5 * v[i-1, j]) + (0.975 * v[i, j+1]) + (1.025 * v[i, j-1]))
    return F

def newton_solver(nx=400, ny=40, tol=1e-10, max_iter=20):
    """Resuelve la ecuación con el método de Newton-Raphson."""
    v = initialize_grid(nx, ny)
    v = apply_boundary_conditions(v)

    for iteration in range(max_iter):
        F = compute_residual(v, nx, ny).flatten()
        J = compute_jacobian(v)  # v ya tiene forma (nx, ny)

        # Analizar diagonales del Jacobiano
        analyze_jacobian_diagonals(J)

        # Resolver el sistema lineal J * Δv = -F
        delta_v = spsolve(J, -F)

        # Actualizar v
        v_new = v.flatten() + delta_v
        v = v_new.reshape((nx, ny))

        # Aplicar condiciones de frontera
        v = apply_boundary_conditions(v)

        # Verificar convergencia
        error = np.linalg.norm(delta_v)
        print(f"Iteración {iteration + 1}, error = {error:.2e}")
        if error < tol:
            print(f"Convergencia alcanzada en {iteration+1} iteraciones.")
            break

    return v, F, J


def compute_jacobian(v):
    """
    Calcula la matriz Jacobiana del sistema F[i,j] para una matriz v[i,j].
    La función F[i,j] es:
        F[i,j] = v[i,j] - (2 * (v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1])) / (8 + v[i+1,j] + v[i-1,j])
    
    Parámetros:
        v: ndarray de tamaño (nx, ny)
    Retorna:
        J: Jacobiano de tamaño (nx*ny, nx*ny)
    """
    nx, ny = v.shape
    N = nx * ny
    J = lil_matrix((N, N)) 
    

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)

            # Frontera: identidad
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                J[k, k] = 1
                continue
        
            elif i == 1 and j == 1:
                # Caso especial para i == 1 y j == 1
                J[k, k] = 1
                J[k, idx(i+1, j)] = -0.125
                J[k, idx(i-1, j)] = 0
                J[k, idx(i, j+1)] = -0.24375 
                J[k, idx(i, j-1)] = 0
                continue
            elif j == 1:
                # Derivadas según el análisis
                J[k, k] = 1
                J[k, idx(i+1, j)] = -0.125
                J[k, idx(i-1, j)] = -0.375
                J[k, idx(i, j+1)] = -0.24375 
                J[k, idx(i, j-1)] = 0
                continue
            elif i == 1:
                # Derivadas según el análisis
                J[k, k] = 1
                J[k, idx(i+1, j)] = -0.125
                J[k, idx(i-1, j)] = 0
                J[k, idx(i, j+1)] = -0.24375 
                J[k, idx(i, j-1)] = -0.25625 
                continue
            elif i == nx - 2:
                # Derivadas según el análisis
                J[k, k] = 1
                J[k, idx(i+1, j)] = 0
                J[k, idx(i-1, j)] = -0.375
                J[k, idx(i, j+1)] = -0.24375 
                J[k, idx(i, j-1)] = -0.25625 
                continue
            elif j == ny - 2:
                # Derivadas según el análisis
                J[k, k] = 1
                J[k, idx(i+1, j)] = -0.125
                J[k, idx(i-1, j)] = -0.375
                J[k, idx(i, j+1)] = 0
                J[k, idx(i, j-1)] = -0.25625 
                continue
            elif i == nx - 2 and j == 1:
                # Caso especial para i == nx - 2 y j == 1
                J[k, k] = 1
                J[k, idx(i+1, j)] = 0
                J[k, idx(i-1, j)] = -0.375
                J[k, idx(i, j+1)] = -0.24375 
                J[k, idx(i, j-1)] = 0
                continue
            elif i == 1 and j == ny - 2:
                # Caso especial para i == 1 y j == ny - 2
                J[k, k] = 1
                J[k, idx(i+1, j)] = -0.125
                J[k, idx(i-1, j)] = 0
                J[k, idx(i, j+1)] = 0
                J[k, idx(i, j-1)] = -0.25625  
                continue
            elif i == nx - 2 and j == ny - 2:
                # Caso especial para i == nx - 2 y j == ny - 2
                J[k, k] = 1
                J[k, idx(i+1, j)] = 0
                J[k, idx(i-1, j)] = -0.375
                J[k, idx(i, j+1)] = 0
                J[k, idx(i, j-1)] = -0.25625  
                continue
            else:
                # Derivadas según el análisis
                J[k, k] = 1
                J[k, idx(i+1, j)] = -0.125  # Vecino derecha
                J[k, idx(i-1, j)] = -0.375  # Vecino izquierda
                J[k, idx(i, j+1)] = -0.24375  # Vecino arriba
                J[k, idx(i, j-1)] = -0.25625  # Vecino abajo


    return J.tocsr()


def smooth_with_spline(v, factor=4):
    """Aplica spline cúbico natural 2D para suavizar la visualización."""
    nx, ny = v.shape
    x = np.arange(nx)
    y = np.arange(ny)
    
    spline = RectBivariateSpline(x, y, v, kx=3, ky=3)

    x_fine = np.linspace(0, nx - 1, nx * factor)
    y_fine = np.linspace(0, ny - 1, ny * factor)

    v_smooth = spline(x_fine, y_fine)
    return v_smooth

def plot_velocity_field(v, smooth=False, factor=4):
    """Grafica la velocidad con opción de suavizado spline."""
    if smooth:
        v = smooth_with_spline(v, factor)

    plt.figure(figsize=(10, 5))
    plt.gca().invert_yaxis()
    im = plt.imshow(v.T, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear', origin='upper')
    plt.colorbar(im, label='Velocidad')
    plt.title("Campo de velocidad en la malla")
    plt.xlabel("i")
    plt.ylabel("j")
    plt.show()

def gauss_seidel_method(J, b, x0=None, tol=1e-10, max_iter=100):
    """
    Método de Gauss-Seidel clásico para Ax = b.
    J: matriz (idealmente dispersa)
    b: vector
    x0: vector inicial
    """
    J = J.tocsr()
    n = J.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    for it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            row_start = J.indptr[i]
            row_end = J.indptr[i+1]
            suma = 0.0
            for idx in range(row_start, row_end):
                j = J.indices[idx]
                if j != i:
                    suma += J.data[idx] * x[j]
            x[i] = (b[i] - suma) / J[i, i]
        error = np.linalg.norm(x - x_old)
        if error < tol:
            break
    return x

def gauss_seidel_newton_like_solver(nx, ny, tol=1e-5, max_iter=5000):
    v = initialize_grid(nx, ny)
    v = apply_boundary_conditions(v)
    for it in range(max_iter):
        F = compute_residual(v, nx, ny).flatten()
        J = compute_jacobian(v)
        # Solo 1 iteración Gauss-Seidel por paso externo
        delta_v = gauss_seidel_method(J, -F, x0=None, tol=1e-8, max_iter=1)
        v_new = v.flatten() + delta_v
        v = v_new.reshape((nx, ny))
        v = apply_boundary_conditions(v)
        error = np.linalg.norm(delta_v, ord=np.inf)
        print(f"Iteración {it+1}, error = {error:.2e}")
        if error < tol:
            print(f"Usando Gauss-seidel convergencia alcanzada en {it+1} iteraciones.")
            break
    return v

velocity_field_gauss_seidel = gauss_seidel_newton_like_solver(400, 40)
plot_velocity_field(velocity_field_gauss_seidel, smooth=True, factor=4)