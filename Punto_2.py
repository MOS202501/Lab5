# Laboratorio 5 MOS: Optimización Multiobjetivo
# Problema 2: Planificación de Rutas de Inspección

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import itertools
from mpl_toolkits.mplot3d import Axes3D
import os

# Configuración para las gráficas
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Definición de conjuntos
num_localidades = 10  # 0-9, donde 0 es el depósito
num_equipos = 3

# Matriz de distancias (simétrica)
# Generamos una matriz de distancias realista simulando coordenadas para las localidades
np.random.seed(42)  # Para reproducibilidad
# Coordenadas aleatorias en un área de 100x100
coords = np.random.rand(num_localidades, 2) * 100

# Calcular distancias euclidianas entre localidades
distancias = np.zeros((num_localidades, num_localidades))
for i in range(num_localidades):
    for j in range(num_localidades):
        distancias[i, j] = np.sqrt(((coords[i, 0] - coords[j, 0]) ** 2) +
                                   ((coords[i, 1] - coords[j, 1]) ** 2))

# Calidad de inspección por localidad
calidad_inspeccion = {
    1: 85,
    2: 92,
    3: 78,
    4: 90,
    5: 82,
    6: 88,
    7: 95,
    8: 75,
    9: 84,
}

# Nivel de riesgo por tramo
# Inicializamos todos los riesgos a un valor medio de 5
riesgo_tramo = {}
for i in range(num_localidades):
    for j in range(num_localidades):
        if i != j:
            riesgo_tramo[(i, j)] = 5

# Actualizamos con los valores específicos dados en el problema
riesgo_especifico = {
    (0, 1): 3, (0, 2): 2, (0, 3): 4, (0, 4): 5, (0, 5): 6,
    (0, 6): 3, (0, 7): 2, (0, 8): 4, (0, 9): 5,
    (2, 8): 9, (2, 9): 8, (3, 4): 5, (4, 9): 7,
    (5, 6): 7, (8, 9): 7
}

# Completamos la matriz de riesgo con valores simétricos para los tramos dados
for (i, j), valor in riesgo_especifico.items():
    riesgo_tramo[(i, j)] = valor
    # Asumimos que el riesgo es el mismo en ambas direcciones
    riesgo_tramo[(j, i)] = valor


def crear_modelo_base():
    """
    Crea y retorna el modelo base con variables, restricciones comunes y sin función objetivo específica.
    """
    modelo = pyo.ConcreteModel()

    # Conjuntos
    modelo.localidades = pyo.Set(initialize=range(num_localidades))
    modelo.localidades_sin_deposito = pyo.Set(
        initialize=range(1, num_localidades))
    modelo.equipos = pyo.Set(initialize=range(num_equipos))

    # Variables de decisión
    # x[i,j,k] = 1 si el equipo k viaja directamente de i a j, 0 en caso contrario
    modelo.x = pyo.Var(modelo.localidades, modelo.localidades, modelo.equipos,
                       domain=pyo.Binary)

    # u[i,k] = variable auxiliar para eliminar subtours para el equipo k
    modelo.u = pyo.Var(modelo.localidades_sin_deposito, modelo.equipos,
                       domain=pyo.NonNegativeIntegers, bounds=(1, num_localidades-1))

    # Restricciones

    # Cada localidad (excepto el depósito) debe ser visitada exactamente una vez por algún equipo
    def restriccion_visita_unica(modelo, j):
        if j == 0:
            return pyo.Constraint.Skip
        return sum(modelo.x[i, j, k] for i in modelo.localidades for k in modelo.equipos if i != j) == 1
    modelo.restriccion_visita_unica = pyo.Constraint(
        modelo.localidades, rule=restriccion_visita_unica)

    # Cada equipo debe salir del depósito
    def restriccion_salida_deposito(modelo, k):
        return sum(modelo.x[0, j, k] for j in modelo.localidades_sin_deposito) <= 1
    modelo.restriccion_salida_deposito = pyo.Constraint(
        modelo.equipos, rule=restriccion_salida_deposito)

    # Cada equipo debe regresar al depósito
    def restriccion_regreso_deposito(modelo, k):
        return sum(modelo.x[i, 0, k] for i in modelo.localidades_sin_deposito) <= 1
    modelo.restriccion_regreso_deposito = pyo.Constraint(
        modelo.equipos, rule=restriccion_regreso_deposito)

    # Balance de flujo: si un equipo llega a una localidad, debe salir de ella
    def restriccion_balance_flujo(modelo, j, k):
        if j == 0:
            return pyo.Constraint.Skip
        return sum(modelo.x[i, j, k] for i in modelo.localidades if i != j) == sum(modelo.x[j, i, k] for i in modelo.localidades if i != j)
    modelo.restriccion_balance_flujo = pyo.Constraint(
        modelo.localidades, modelo.equipos, rule=restriccion_balance_flujo)

    # Eliminación de subtours mediante restricciones MTZ (Miller-Tucker-Zemlin)
    def restriccion_mtz(modelo, i, j, k):
        if i == 0 or j == 0 or i == j:
            return pyo.Constraint.Skip
        return modelo.u[i, k] - modelo.u[j, k] + (num_localidades - 1) * modelo.x[i, j, k] <= num_localidades - 2
    modelo.restriccion_mtz = pyo.Constraint(
        modelo.localidades_sin_deposito, modelo.localidades_sin_deposito, modelo.equipos, rule=restriccion_mtz)

    # Un equipo solo puede usarse si sale del depósito
    def restriccion_uso_equipo(modelo, k):
        return sum(modelo.x[i, j, k] for i in modelo.localidades for j in modelo.localidades if i != j) <= (num_localidades - 1) * sum(modelo.x[0, j, k] for j in modelo.localidades_sin_deposito)
    modelo.restriccion_uso_equipo = pyo.Constraint(
        modelo.equipos, rule=restriccion_uso_equipo)

    return modelo


def funcion_objetivo_distancia(modelo):
    """Define la función objetivo para minimizar la distancia total recorrida."""
    return sum(distancias[i, j] * modelo.x[i, j, k]
               for i in modelo.localidades for j in modelo.localidades for k in modelo.equipos
               if i != j)


def funcion_objetivo_calidad(modelo):
    """Define la función objetivo para maximizar la calidad de inspección acumulada."""
    return sum(calidad_inspeccion.get(j, 0) * sum(modelo.x[i, j, k] for i in modelo.localidades if i != j)
               for j in range(1, num_localidades) for k in modelo.equipos)


def funcion_objetivo_riesgo(modelo):
    """Define la función objetivo para minimizar el nivel de riesgo de la ruta."""
    return sum(riesgo_tramo.get((i, j), 5) * modelo.x[i, j, k]
               for i in modelo.localidades for j in modelo.localidades for k in modelo.equipos
               if i != j)


def calcular_valores_extremos():
    """
    Calcula los valores extremos para cada función objetivo optimizando cada una por separado.
    """
    print("Calculando valores extremos para cada objetivo...")

    resultados = {}

    # Optimizar para minimizar distancia
    modelo = crear_modelo_base()
    modelo.objetivo = pyo.Objective(
        rule=funcion_objetivo_distancia, sense=pyo.minimize)

    solver = SolverFactory('glpk')
    resultados_solver = solver.solve(modelo, tee=False)

    if resultados_solver.solver.termination_condition == TerminationCondition.optimal:
        min_distancia = pyo.value(modelo.objetivo)
        # También calculamos los valores de los otros objetivos para esta solución
        calidad_en_min_distancia = sum(calidad_inspeccion.get(j, 0) * sum(pyo.value(modelo.x[i, j, k]) for i in modelo.localidades if i != j)
                                       for j in range(1, num_localidades) for k in modelo.equipos)
        riesgo_en_min_distancia = sum(riesgo_tramo.get((i, j), 5) * pyo.value(modelo.x[i, j, k])
                                      for i in modelo.localidades for j in modelo.localidades for k in modelo.equipos if i != j)

        print(f"Mínima distancia total: {min_distancia:.2f}")
        print(
            f"Calidad en solución de mínima distancia: {calidad_en_min_distancia:.2f}")
        print(
            f"Riesgo en solución de mínima distancia: {riesgo_en_min_distancia:.2f}")
    else:
        print("No se pudo encontrar una solución óptima para minimizar distancia.")
        min_distancia = 1000  # Valor por defecto
        calidad_en_min_distancia = 0
        riesgo_en_min_distancia = 100

    # Optimizar para maximizar calidad
    modelo = crear_modelo_base()
    modelo.objetivo = pyo.Objective(
        rule=funcion_objetivo_calidad, sense=pyo.maximize)

    resultados_solver = solver.solve(modelo, tee=False)

    if resultados_solver.solver.termination_condition == TerminationCondition.optimal:
        max_calidad = pyo.value(modelo.objetivo)
        # Calculamos los valores de los otros objetivos
        distancia_en_max_calidad = sum(distancias[i, j] * pyo.value(modelo.x[i, j, k])
                                       for i in modelo.localidades for j in modelo.localidades for k in modelo.equipos if i != j)
        riesgo_en_max_calidad = sum(riesgo_tramo.get((i, j), 5) * pyo.value(modelo.x[i, j, k])
                                    for i in modelo.localidades for j in modelo.localidades for k in modelo.equipos if i != j)

        print(f"Máxima calidad acumulada: {max_calidad:.2f}")
        print(
            f"Distancia en solución de máxima calidad: {distancia_en_max_calidad:.2f}")
        print(
            f"Riesgo en solución de máxima calidad: {riesgo_en_max_calidad:.2f}")
    else:
        print("No se pudo encontrar una solución óptima para maximizar calidad.")
        max_calidad = 500  # Valor por defecto
        distancia_en_max_calidad = 1000
        riesgo_en_max_calidad = 100

    # Optimizar para minimizar riesgo
    modelo = crear_modelo_base()
    modelo.objetivo = pyo.Objective(
        rule=funcion_objetivo_riesgo, sense=pyo.minimize)

    resultados_solver = solver.solve(modelo, tee=False)

    if resultados_solver.solver.termination_condition == TerminationCondition.optimal:
        min_riesgo = pyo.value(modelo.objetivo)
        # Calculamos los valores de los otros objetivos
        distancia_en_min_riesgo = sum(distancias[i, j] * pyo.value(modelo.x[i, j, k])
                                      for i in modelo.localidades for j in modelo.localidades for k in modelo.equipos if i != j)
        calidad_en_min_riesgo = sum(calidad_inspeccion.get(j, 0) * sum(pyo.value(modelo.x[i, j, k]) for i in modelo.localidades if i != j)
                                    for j in range(1, num_localidades) for k in modelo.equipos)

        print(f"Mínimo riesgo total: {min_riesgo:.2f}")
        print(
            f"Distancia en solución de mínimo riesgo: {distancia_en_min_riesgo:.2f}")
        print(
            f"Calidad en solución de mínimo riesgo: {calidad_en_min_riesgo:.2f}")
    else:
        print("No se pudo encontrar una solución óptima para minimizar riesgo.")
        min_riesgo = 50  # Valor por defecto
        distancia_en_min_riesgo = 1000
        calidad_en_min_riesgo = 0

    # Calcular valores máximos para los objetivos de minimización
    max_distancia = max(distancia_en_max_calidad, distancia_en_min_riesgo)
    max_riesgo = max(riesgo_en_min_distancia, riesgo_en_max_calidad)

    # Calcular valor mínimo para el objetivo de maximización
    min_calidad = min(calidad_en_min_distancia, calidad_en_min_riesgo)

    # Asegurar que hay una diferencia entre valores mínimos y máximos
    if abs(max_calidad - min_calidad) < 1e-10:
        print("Advertencia: La calidad es constante en todas las soluciones.")
        # Si son iguales, añadimos un pequeño delta para evitar división por cero
        max_calidad = min_calidad * 1.05  # Incrementamos en un 5%

    if abs(max_distancia - min_distancia) < 1e-10:
        max_distancia = min_distancia * 1.05

    if abs(max_riesgo - min_riesgo) < 1e-10:
        max_riesgo = min_riesgo * 1.05

    resultados = {
        'min_distancia': min_distancia,
        'max_distancia': max_distancia,
        'min_calidad': min_calidad,
        'max_calidad': max_calidad,
        'min_riesgo': min_riesgo,
        'max_riesgo': max_riesgo
    }

    return resultados


def metodo_suma_ponderada(num_puntos=7, valores_extremos=None):
    """
    Implementa el método de la suma ponderada para generar el frente de Pareto.

    Args:
        num_puntos: número de combinaciones de pesos a usar
        valores_extremos: diccionario con valores extremos para normalización

    Returns:
        DataFrame con los resultados para diferentes combinaciones de pesos
    """
    if valores_extremos is None:
        valores_extremos = calcular_valores_extremos()

    min_distancia = valores_extremos['min_distancia']
    max_distancia = valores_extremos['max_distancia']
    min_calidad = valores_extremos['min_calidad']
    max_calidad = valores_extremos['max_calidad']
    min_riesgo = valores_extremos['min_riesgo']
    max_riesgo = valores_extremos['max_riesgo']

    # Generar combinaciones de pesos
    # Para 3 objetivos, necesitamos combinar los pesos para sumar 1
    # Usando una lista predefinida para controlar mejor las combinaciones
    pesos = [
        (0.0, 0.0, 1.0),  # Solo riesgo
        (0.0, 0.33, 0.67),  # Calidad y riesgo, énfasis en riesgo
        (0.0, 0.67, 0.33),  # Calidad y riesgo, énfasis en calidad
        (0.0, 1.0, 0.0),   # Solo calidad
        (0.33, 0.0, 0.67),  # Distancia y riesgo, énfasis en riesgo
        (0.33, 0.33, 0.33),  # Equilibrado
        (0.33, 0.67, 0.0),  # Distancia y calidad, énfasis en calidad
        (0.67, 0.0, 0.33),  # Distancia y riesgo, énfasis en distancia
        (0.67, 0.33, 0.0),  # Distancia y calidad, énfasis en distancia
        (1.0, 0.0, 0.0)     # Solo distancia
    ]

    resultados = []

    print(
        f"Generando frente de Pareto con {len(pesos)} combinaciones de pesos...")

    for idx, (alpha, beta, gamma) in enumerate(pesos):
        modelo = crear_modelo_base()

        # Función objetivo ponderada normalizada
        def objetivo_ponderado(modelo):
            # Evitar división por cero
            distancia_norm = (funcion_objetivo_distancia(
                modelo) - min_distancia) / max(max_distancia - min_distancia, 1e-10)

            # Para calidad, manejamos caso especial si min_calidad == max_calidad
            if abs(max_calidad - min_calidad) < 1e-10:
                calidad_norm = 0.5  # Valor constante ya que calidad es constante
            else:
                calidad_norm = (funcion_objetivo_calidad(
                    modelo) - min_calidad) / (max_calidad - min_calidad)

            riesgo_norm = (funcion_objetivo_riesgo(modelo) -
                           min_riesgo) / max(max_riesgo - min_riesgo, 1e-10)

            # Para calidad, invertimos el signo ya que queremos maximizar
            return alpha * distancia_norm - beta * calidad_norm + gamma * riesgo_norm

        modelo.objetivo = pyo.Objective(
            rule=objetivo_ponderado, sense=pyo.minimize)

        solver = SolverFactory('glpk')
        resultados_solver = solver.solve(modelo, tee=False)

        if resultados_solver.solver.termination_condition == TerminationCondition.optimal:
            # Calcular valores reales de los objetivos
            distancia = pyo.value(funcion_objetivo_distancia(modelo))
            calidad = pyo.value(funcion_objetivo_calidad(modelo))
            riesgo = pyo.value(funcion_objetivo_riesgo(modelo))

            print(
                f"Combinación {idx+1}/{len(pesos)}: Pesos = ({alpha:.2f}, {beta:.2f}, {gamma:.2f})")
            print(
                f"  Distancia = {distancia:.2f}, Calidad = {calidad:.2f}, Riesgo = {riesgo:.2f}")

            # Almacenar los resultados
            ruta_por_equipo = {}
            for k in range(num_equipos):
                ruta = []
                actual = 0  # Empezamos en el depósito
                visitado = set([0])

                # Si este equipo no se usa (no sale del depósito), continuamos
                if sum(pyo.value(modelo.x[0, j, k]) for j in range(1, num_localidades)) < 0.5:
                    continue

                # Reconstruir la ruta
                while len(visitado) < num_localidades:
                    for j in range(num_localidades):
                        if j != actual and pyo.value(modelo.x[actual, j, k]) > 0.5:
                            ruta.append((actual, j))
                            visitado.add(j)
                            actual = j
                            break
                    # Si no encontramos un siguiente nodo o si volvimos al depósito, terminamos
                    if actual == 0 or len(ruta) == 0 or ruta[-1][1] == 0:
                        break

                # Si no terminamos en el depósito, añadimos el último tramo
                if actual != 0:
                    ruta.append((actual, 0))

                ruta_por_equipo[k] = ruta

            resultados.append({
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'distancia': distancia,
                'calidad': calidad,
                'riesgo': riesgo,
                'rutas': ruta_por_equipo
            })
        else:
            print(
                f"No se pudo encontrar una solución óptima para los pesos ({alpha:.2f}, {beta:.2f}, {gamma:.2f})")

    return pd.DataFrame(resultados)


def metodo_epsilon_constraint(num_puntos=3, valores_extremos=None):
    """
    Implementa el método ε-constraint para generar el frente de Pareto.

    Args:
        num_puntos: número de puntos para generar en el frente de Pareto
        valores_extremos: diccionario con valores extremos para normalización

    Returns:
        DataFrame con los resultados de las diferentes restricciones ε
    """
    if valores_extremos is None:
        valores_extremos = calcular_valores_extremos()

    min_distancia = valores_extremos['min_distancia']
    max_distancia = valores_extremos['max_distancia']
    min_calidad = valores_extremos['min_calidad']
    max_calidad = valores_extremos['max_calidad']
    min_riesgo = valores_extremos['min_riesgo']
    max_riesgo = valores_extremos['max_riesgo']

    # Generar diferentes valores de epsilon para distancia y riesgo
    epsilons_distancia = np.linspace(min_distancia, max_distancia, num_puntos)
    epsilons_riesgo = np.linspace(min_riesgo, max_riesgo, num_puntos)

    resultados = []

    print(
        f"Generando frente de Pareto con {len(epsilons_distancia) * len(epsilons_riesgo)} combinaciones de epsilons...")

    contador = 0
    total_combinaciones = len(epsilons_distancia) * len(epsilons_riesgo)

    for epsilon_distancia in epsilons_distancia:
        for epsilon_riesgo in epsilons_riesgo:
            contador += 1

            modelo = crear_modelo_base()

            # Función objetivo: maximizar calidad
            modelo.objetivo = pyo.Objective(
                rule=funcion_objetivo_calidad, sense=pyo.maximize)

            # Restricciones epsilon para distancia y riesgo
            modelo.restriccion_epsilon_distancia = pyo.Constraint(
                expr=funcion_objetivo_distancia(modelo) <= epsilon_distancia)

            modelo.restriccion_epsilon_riesgo = pyo.Constraint(
                expr=funcion_objetivo_riesgo(modelo) <= epsilon_riesgo)

            solver = SolverFactory('glpk')
            resultados_solver = solver.solve(modelo, tee=False)

            if resultados_solver.solver.termination_condition == TerminationCondition.optimal:
                # Calcular valores reales de los objetivos
                distancia = pyo.value(funcion_objetivo_distancia(modelo))
                calidad = pyo.value(funcion_objetivo_calidad(modelo))
                riesgo = pyo.value(funcion_objetivo_riesgo(modelo))

                print(
                    f"Combinación {contador}/{total_combinaciones}: Epsilon Distancia = {epsilon_distancia:.2f}, Epsilon Riesgo = {epsilon_riesgo:.2f}")
                print(
                    f"  Distancia = {distancia:.2f}, Calidad = {calidad:.2f}, Riesgo = {riesgo:.2f}")

                # Almacenar los resultados
                ruta_por_equipo = {}
                for k in range(num_equipos):
                    ruta = []
                    actual = 0  # Empezamos en el depósito
                    visitado = set([0])

                    # Si este equipo no se usa (no sale del depósito), continuamos
                    if sum(pyo.value(modelo.x[0, j, k]) for j in range(1, num_localidades)) < 0.5:
                        continue

                    # Reconstruir la ruta
                    while len(visitado) < num_localidades:
                        for j in range(num_localidades):
                            if j != actual and pyo.value(modelo.x[actual, j, k]) > 0.5:
                                ruta.append((actual, j))
                                visitado.add(j)
                                actual = j
                                break
                        # Si no encontramos un siguiente nodo o si volvimos al depósito, terminamos
                        if actual == 0 or len(ruta) == 0 or ruta[-1][1] == 0:
                            break

                    # Si no terminamos en el depósito, añadimos el último tramo
                    if actual != 0:
                        ruta.append((actual, 0))

                    ruta_por_equipo[k] = ruta

                resultados.append({
                    'epsilon_distancia': epsilon_distancia,
                    'epsilon_riesgo': epsilon_riesgo,
                    'distancia': distancia,
                    'calidad': calidad,
                    'riesgo': riesgo,
                    'rutas': ruta_por_equipo
                })
            else:
                print(
                    f"No se pudo encontrar una solución óptima para epsilons ({epsilon_distancia:.2f}, {epsilon_riesgo:.2f})")

    return pd.DataFrame(resultados)


def es_pareto_optimo(df):
    """
    Identifica las soluciones Pareto-óptimas en un DataFrame.

    Args:
        df: DataFrame con columnas 'distancia', 'calidad', 'riesgo'

    Returns:
        Serie booleana indicando si cada fila es Pareto-óptima
    """
    if len(df) == 0:
        return pd.Series([], dtype=bool)

    es_optimo = np.ones(len(df), dtype=bool)

    for i, row_i in df.iterrows():
        # Para cada solución, verificamos si es dominada por alguna otra
        for j, row_j in df.iterrows():
            if i != j:
                # Menor distancia, mayor calidad, menor riesgo es mejor
                if (row_j['distancia'] <= row_i['distancia'] and
                    row_j['calidad'] >= row_i['calidad'] and
                    row_j['riesgo'] <= row_i['riesgo'] and
                    (row_j['distancia'] < row_i['distancia'] or
                     row_j['calidad'] > row_i['calidad'] or
                     row_j['riesgo'] < row_i['riesgo'])):
                    es_optimo[i] = False
                    break

    return es_optimo


def graficar_frente_pareto_3d(df, método):
    """
    Genera una visualización 3D del frente de Pareto.

    Args:
        df: DataFrame con los resultados
        método: nombre del método utilizado
    """
    # Si el DataFrame está vacío, mostramos un mensaje y salimos
    if len(df) == 0:
        print(f"No hay soluciones para graficar del método {método}")
        return

    # Identificar soluciones Pareto-óptimas
    df['es_pareto_optimo'] = es_pareto_optimo(df)

    # Verificar si hay variación en los objetivos
    calidad_constante = df['calidad'].std() < 1e-10

    # Crear figura 3D si hay variación en todos los objetivos
    if not calidad_constante:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Graficamos todas las soluciones
        scatter = ax.scatter(df['distancia'], df['calidad'], df['riesgo'],
                             c=['red' if optimo else 'blue' for optimo in df['es_pareto_optimo']],
                             s=50, alpha=0.6)

        # Destacamos las soluciones Pareto-óptimas
        df_pareto = df[df['es_pareto_optimo']]
        if len(df_pareto) > 0:
            ax.scatter(df_pareto['distancia'],
                       df_pareto['calidad'],
                       df_pareto['riesgo'],
                       c='red', s=100, label='Pareto-óptimas')

        ax.set_xlabel('Distancia Total')
        ax.set_ylabel('Calidad de Inspección')
        ax.set_zlabel('Riesgo Total')
        ax.set_title(f'Frente de Pareto 3D - Método {método}')

        # Añadir leyenda
        ax.legend()

        plt.savefig(f'img/punto_2/frente_pareto_3d_{método}.png', dpi=300)
        plt.show()
    else:
        # Si la calidad es constante, hacemos un gráfico 2D de distancia vs riesgo
        plt.figure(figsize=(12, 8))

        # Graficar todas las soluciones
        plt.scatter(df['distancia'], df['riesgo'],
                    c=['red' if optimo else 'blue' for optimo in df['es_pareto_optimo']],
                    s=70, alpha=0.7)

        # Destacar las soluciones Pareto-óptimas
        df_pareto = df[df['es_pareto_optimo']]
        if len(df_pareto) > 0:
            plt.scatter(df_pareto['distancia'], df_pareto['riesgo'],
                        c='red', s=120, edgecolors='black', label='Pareto-óptimas')

        plt.xlabel('Distancia Total')
        plt.ylabel('Riesgo Total')
        plt.title(
            f'Frente de Pareto 2D (Calidad constante = {df["calidad"].mean():.2f}) - Método {método}')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'img/punto_2/frente_pareto_2d_{método}.png', dpi=300)
        plt.show()

    # También graficamos proyecciones 2D para mejor visualización
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Distancia vs Calidad
    axs[0].scatter(df['distancia'], df['calidad'],
                   c=['red' if optimo else 'blue' for optimo in df['es_pareto_optimo']],
                   s=50, alpha=0.6)
    axs[0].set_xlabel('Distancia Total')
    axs[0].set_ylabel('Calidad de Inspección')
    axs[0].set_title('Distancia vs Calidad')

    # Distancia vs Riesgo
    axs[1].scatter(df['distancia'], df['riesgo'],
                   c=['red' if optimo else 'blue' for optimo in df['es_pareto_optimo']],
                   s=50, alpha=0.6)
    axs[1].set_xlabel('Distancia Total')
    axs[1].set_ylabel('Riesgo Total')
    axs[1].set_title('Distancia vs Riesgo')

    # Calidad vs Riesgo
    axs[2].scatter(df['calidad'], df['riesgo'],
                   c=['red' if optimo else 'blue' for optimo in df['es_pareto_optimo']],
                   s=50, alpha=0.6)
    axs[2].set_xlabel('Calidad de Inspección')
    axs[2].set_ylabel('Riesgo Total')
    axs[2].set_title('Calidad vs Riesgo')

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'img/punto_2/proyecciones_2d_{método}.png', dpi=300)
    plt.show()


def analizar_tradeoffs(df_pareto):
    """
    Analiza los trade-offs entre los diferentes objetivos.

    Args:
        df_pareto: DataFrame con soluciones Pareto-óptimas
    """
    if len(df_pareto) <= 1:
        print("\nNo hay suficientes soluciones Pareto-óptimas para analizar trade-offs.")
        return

    # Verificar si hay variación en la calidad
    calidad_constante = df_pareto['calidad'].std() < 1e-10

    # Calcular correlaciones entre objetivos de manera segura
    try:
        corr_dist_calidad = df_pareto['distancia'].corr(df_pareto['calidad'])
    except:
        corr_dist_calidad = float('nan')

    try:
        corr_dist_riesgo = df_pareto['distancia'].corr(df_pareto['riesgo'])
    except:
        corr_dist_riesgo = float('nan')

    try:
        corr_calidad_riesgo = df_pareto['calidad'].corr(df_pareto['riesgo'])
    except:
        corr_calidad_riesgo = float('nan')

    print("\nAnálisis de Trade-offs entre Objetivos:")
    print(
        f"Correlación Distancia-Calidad: {corr_dist_calidad if not np.isnan(corr_dist_calidad) else 'No aplicable (calidad constante)'}")
    print(
        f"Correlación Distancia-Riesgo: {corr_dist_riesgo if not np.isnan(corr_dist_riesgo) else 'No aplicable'}")
    print(
        f"Correlación Calidad-Riesgo: {corr_calidad_riesgo if not np.isnan(corr_calidad_riesgo) else 'No aplicable (calidad constante)'}")

    # Analizar tasas de cambio - de manera segura
    print("\nTasas de Cambio entre Objetivos:")

    # Ordenar por distancia para analizar cómo cambian los otros objetivos
    if len(df_pareto) > 1:
        df_sorted_dist = df_pareto.sort_values(
            'distancia').reset_index(drop=True)
        delta_dist = df_sorted_dist['distancia'].diff()

        # Verificar si hay variación en calidad
        if not calidad_constante:
            try:
                delta_calidad_por_dist = df_sorted_dist['calidad'].diff(
                ) / delta_dist
                print("\nCambio promedio en Calidad por unidad de Distancia:")
                print(
                    f"  {delta_calidad_por_dist.mean():.4f} unidades de calidad por unidad de distancia")
            except:
                print("\nCambio promedio en Calidad por unidad de Distancia:")
                print("  No aplicable (calidad constante o división por cero)")
        else:
            print("\nCambio promedio en Calidad por unidad de Distancia:")
            print("  No aplicable (calidad constante en todas las soluciones)")

        try:
            delta_riesgo_por_dist = df_sorted_dist['riesgo'].diff(
            ) / delta_dist
            delta_riesgo_por_dist = delta_riesgo_por_dist.dropna()
            if len(delta_riesgo_por_dist) > 0:
                print("Cambio promedio en Riesgo por unidad de Distancia:")
                print(
                    f"  {delta_riesgo_por_dist.mean():.4f} unidades de riesgo por unidad de distancia")
            else:
                print("Cambio promedio en Riesgo por unidad de Distancia:")
                print("  No hay datos suficientes para el cálculo")
        except Exception as e:
            print("Cambio promedio en Riesgo por unidad de Distancia:")
            print(f"  No aplicable ({str(e)})")

        # Verificar si hay variación en calidad antes de calcular
        if not calidad_constante:
            try:
                # Ordenar por calidad para analizar cómo cambia el riesgo
                df_sorted_calidad = df_pareto.sort_values(
                    'calidad').reset_index(drop=True)
                delta_calidad = df_sorted_calidad['calidad'].diff()
                delta_riesgo_por_calidad = df_sorted_calidad['riesgo'].diff(
                ) / delta_calidad
                delta_riesgo_por_calidad = delta_riesgo_por_calidad.dropna()

                if len(delta_riesgo_por_calidad) > 0:
                    print("\nCambio promedio en Riesgo por unidad de Calidad:")
                    print(
                        f"  {delta_riesgo_por_calidad.mean():.4f} unidades de riesgo por unidad de calidad")
                else:
                    print("\nCambio promedio en Riesgo por unidad de Calidad:")
                    print("  No hay datos suficientes para el cálculo")
            except Exception as e:
                print("\nCambio promedio en Riesgo por unidad de Calidad:")
                print(f"  No aplicable ({str(e)})")
        else:
            print("\nCambio promedio en Riesgo por unidad de Calidad:")
            print("  No aplicable (calidad constante en todas las soluciones)")
    else:
        print("  Insuficientes datos para calcular tasas de cambio (se necesitan al menos 2 soluciones)")

    # Visualizar los trade-offs
    try:
        plt.figure(figsize=(12, 8))

        if calidad_constante:
            # Si la calidad es constante, visualizar solo distancia vs riesgo
            scatter = plt.scatter(df_pareto['distancia'], df_pareto['riesgo'],
                                  s=80, alpha=0.7,
                                  c=df_pareto.index, cmap='viridis')

            plt.colorbar(scatter, label='Índice de solución')
            plt.xlabel('Distancia Total')
            plt.ylabel('Riesgo Total')
            plt.title('Trade-off entre Distancia y Riesgo (Calidad constante)')
        else:
            # Distancia vs Calidad con tamaño de punto proporcional al Riesgo
            scatter = plt.scatter(df_pareto['distancia'], df_pareto['calidad'],
                                  s=df_pareto['riesgo']*10, alpha=0.7,
                                  c=df_pareto.index, cmap='viridis')

            plt.colorbar(scatter, label='Índice de solución')
            plt.xlabel('Distancia Total')
            plt.ylabel('Calidad de Inspección')
            plt.title('Trade-offs entre Objetivos (tamaño = riesgo)')

        plt.grid(True)

        plt.savefig('img/punto_2/tradeoffs_analisis.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error al visualizar trade-offs: {e}")


def identificar_mejor_solucion_compromiso(df_pareto, valores_extremos):
    """
    Identifica la mejor solución de compromiso basada en diferentes criterios.

    Args:
        df_pareto: DataFrame con soluciones Pareto-óptimas
        valores_extremos: diccionario con valores extremos para normalización

    Returns:
        Índice de la mejor solución de compromiso
    """
    print("\nIdentificación de la Mejor Solución de Compromiso:")

    if len(df_pareto) == 0:
        print("No hay soluciones Pareto-óptimas para analizar.")
        return 0

    if len(df_pareto) == 1:
        print("Solo hay una solución Pareto-óptima disponible.")
        return 0

    # Normalizar objetivos
    df_norm = df_pareto.copy()

    min_distancia = valores_extremos['min_distancia']
    max_distancia = valores_extremos['max_distancia']
    min_calidad = valores_extremos['min_calidad']
    max_calidad = valores_extremos['max_calidad']
    min_riesgo = valores_extremos['min_riesgo']
    max_riesgo = valores_extremos['max_riesgo']

    # Evitar división por cero
    distancia_range = max(max_distancia - min_distancia, 1e-10)
    calidad_range = max(max_calidad - min_calidad, 1e-10)
    riesgo_range = max(max_riesgo - min_riesgo, 1e-10)

    df_norm['distancia_norm'] = (
        df_pareto['distancia'] - min_distancia) / distancia_range
    df_norm['calidad_norm'] = (
        df_pareto['calidad'] - min_calidad) / calidad_range
    df_norm['riesgo_norm'] = (df_pareto['riesgo'] - min_riesgo) / riesgo_range

    # Verificar si hay variación en la calidad
    calidad_constante = df_pareto['calidad'].std() < 1e-10

    # Método 1: Distancia al punto ideal
    # El punto ideal normalizado sería (0, 1, 0) [mínima distancia, máxima calidad, mínimo riesgo]
    try:
        df_norm['distancia_punto_ideal'] = np.sqrt(
            (df_norm['distancia_norm'] - 0)**2 +
            (df_norm['calidad_norm'] - 1)**2 +
            (df_norm['riesgo_norm'] - 0)**2
        )

        idx_min_distancia = df_norm['distancia_punto_ideal'].idxmin()
        if pd.isna(idx_min_distancia):
            idx_min_distancia = 0
    except Exception as e:
        print(f"Error al calcular distancia al punto ideal: {e}")
        idx_min_distancia = 0

    # Método 2: Método TOPSIS
    try:
        # Solución ideal y anti-ideal
        # [distancia_norm, calidad_norm, riesgo_norm]
        ideal_solution = [0, 1, 0]
        anti_ideal_solution = [1, 0, 1]

        # Calcular distancias
        df_norm['dist_to_ideal'] = np.sqrt(
            (df_norm['distancia_norm'] - ideal_solution[0])**2 +
            (df_norm['calidad_norm'] - ideal_solution[1])**2 +
            (df_norm['riesgo_norm'] - ideal_solution[2])**2
        )

        df_norm['dist_to_anti_ideal'] = np.sqrt(
            (df_norm['distancia_norm'] - anti_ideal_solution[0])**2 +
            (df_norm['calidad_norm'] - anti_ideal_solution[1])**2 +
            (df_norm['riesgo_norm'] - anti_ideal_solution[2])**2
        )

        # Calcular ratio de similitud relativa
        eps = 1e-10  # pequeño valor para evitar división por cero
        df_norm['topsis_score'] = df_norm['dist_to_anti_ideal'] / \
            (df_norm['dist_to_ideal'] + df_norm['dist_to_anti_ideal'] + eps)

        idx_max_topsis = df_norm['topsis_score'].idxmax()
        if pd.isna(idx_max_topsis):
            idx_max_topsis = 0
    except Exception as e:
        print(f"Error al calcular TOPSIS: {e}")
        idx_max_topsis = 0

    # Método 3: Suma ponderada equilibrada (pesos iguales)
    try:
        df_norm['suma_ponderada_eq'] = (1/3) * (1 - df_norm['distancia_norm']) + (
            1/3) * df_norm['calidad_norm'] + (1/3) * (1 - df_norm['riesgo_norm'])

        idx_max_suma_eq = df_norm['suma_ponderada_eq'].idxmax()
        if pd.isna(idx_max_suma_eq):
            idx_max_suma_eq = 0
    except Exception as e:
        print(f"Error al calcular suma ponderada: {e}")
        idx_max_suma_eq = 0

    # Presentar resultados
    try:
        print(f"\nMétodo de Distancia al Punto Ideal:")
        print(f"  Mejor solución: Índice {idx_min_distancia}")
        print(
            f"  Distancia: {df_pareto.loc[idx_min_distancia, 'distancia']:.2f}")
        print(f"  Calidad: {df_pareto.loc[idx_min_distancia, 'calidad']:.2f}")
        print(f"  Riesgo: {df_pareto.loc[idx_min_distancia, 'riesgo']:.2f}")

        print(f"\nMétodo TOPSIS:")
        print(f"  Mejor solución: Índice {idx_max_topsis}")
        print(f"  Distancia: {df_pareto.loc[idx_max_topsis, 'distancia']:.2f}")
        print(f"  Calidad: {df_pareto.loc[idx_max_topsis, 'calidad']:.2f}")
        print(f"  Riesgo: {df_pareto.loc[idx_max_topsis, 'riesgo']:.2f}")

        print(f"\nMétodo de Suma Ponderada Equilibrada:")
        print(f"  Mejor solución: Índice {idx_max_suma_eq}")
        print(
            f"  Distancia: {df_pareto.loc[idx_max_suma_eq, 'distancia']:.2f}")
        print(f"  Calidad: {df_pareto.loc[idx_max_suma_eq, 'calidad']:.2f}")
        print(f"  Riesgo: {df_pareto.loc[idx_max_suma_eq, 'riesgo']:.2f}")
    except Exception as e:
        print(f"Error al presentar resultados: {e}")

    # Visualizar soluciones destacadas
    try:
        if calidad_constante:
            # Visualización 2D si la calidad es constante
            plt.figure(figsize=(12, 8))

            # Graficar todas las soluciones
            plt.scatter(df_pareto['distancia'], df_pareto['riesgo'],
                        c='blue', s=50, alpha=0.4, label='Soluciones Pareto-óptimas')

            # Destacar las mejores soluciones según los diferentes métodos
            plt.scatter(df_pareto.loc[idx_min_distancia, 'distancia'],
                        df_pareto.loc[idx_min_distancia, 'riesgo'],
                        c='red', s=200, label='Mejor por Distancia al Punto Ideal', marker='*')

            plt.scatter(df_pareto.loc[idx_max_topsis, 'distancia'],
                        df_pareto.loc[idx_max_topsis, 'riesgo'],
                        c='green', s=200, label='Mejor por TOPSIS', marker='^')

            plt.scatter(df_pareto.loc[idx_max_suma_eq, 'distancia'],
                        df_pareto.loc[idx_max_suma_eq, 'riesgo'],
                        c='orange', s=200, label='Mejor por Suma Ponderada', marker='s')

            plt.xlabel('Distancia Total')
            plt.ylabel('Riesgo Total')
            plt.title('Mejores Soluciones de Compromiso (Calidad constante)')

            # Añadir leyenda
            plt.legend()
        else:
            # Visualización 3D si hay variación en los tres objetivos
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Graficar todas las soluciones Pareto-óptimas
            ax.scatter(df_pareto['distancia'], df_pareto['calidad'], df_pareto['riesgo'],
                       c='blue', s=50, alpha=0.4, label='Soluciones Pareto-óptimas')

            # Destacar las mejores soluciones según los diferentes métodos
            ax.scatter(df_pareto.loc[idx_min_distancia, 'distancia'],
                       df_pareto.loc[idx_min_distancia, 'calidad'],
                       df_pareto.loc[idx_min_distancia, 'riesgo'],
                       c='red', s=200, label='Mejor por Distancia al Punto Ideal', marker='*')

            ax.scatter(df_pareto.loc[idx_max_topsis, 'distancia'],
                       df_pareto.loc[idx_max_topsis, 'calidad'],
                       df_pareto.loc[idx_max_topsis, 'riesgo'],
                       c='green', s=200, label='Mejor por TOPSIS', marker='^')

            ax.scatter(df_pareto.loc[idx_max_suma_eq, 'distancia'],
                       df_pareto.loc[idx_max_suma_eq, 'calidad'],
                       df_pareto.loc[idx_max_suma_eq, 'riesgo'],
                       c='orange', s=200, label='Mejor por Suma Ponderada', marker='s')

            ax.set_xlabel('Distancia Total')
            ax.set_ylabel('Calidad de Inspección')
            ax.set_zlabel('Riesgo Total')
            ax.set_title('Mejores Soluciones de Compromiso')

            # Añadir leyenda
            ax.legend()

        plt.savefig('img/punto_2/mejores_soluciones_compromiso.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error al visualizar soluciones: {e}")

    # Retornamos la solución por TOPSIS o una alternativa si hay error
    if pd.isna(idx_max_topsis):
        return 0 if len(df_pareto) > 0 else 0
    return idx_max_topsis


def visualizar_solucion(solucion, titulo):
    """
    Visualiza la solución de ruteo seleccionada.

    Args:
        solucion: diccionario con los datos de la solución
        titulo: título para la visualización
    """
    try:
        plt.figure(figsize=(12, 10))

        # Graficar las localidades
        plt.scatter(coords[:, 0], coords[:, 1], s=150,
                    c='lightgray', edgecolors='black')

        # Etiquetar cada localidad
        for i in range(num_localidades):
            label = f"{i}" if i > 0 else f"{i} (Depósito)"
            plt.annotate(label, (coords[i, 0], coords[i, 1]),
                         fontsize=12, ha='center', va='center')

        # Mostrar la calidad de inspección para cada localidad
        for i in range(1, num_localidades):
            plt.annotate(f"C:{calidad_inspeccion[i]}",
                         (coords[i, 0], coords[i, 1] - 5),
                         fontsize=10, ha='center', va='top')

        # Graficar las rutas por equipo con diferentes colores
        colores = ['red', 'green', 'blue', 'purple', 'orange']

        rutas = solucion['rutas']

        for equipo, ruta in rutas.items():
            color = colores[equipo % len(colores)]

            # Dibujar cada segmento de la ruta
            for origen, destino in ruta:
                plt.plot([coords[origen, 0], coords[destino, 0]],
                         [coords[origen, 1], coords[destino, 1]],
                         color=color, linewidth=2, marker='o', markersize=8,
                         label=f"Equipo {equipo}" if (origen, destino) == ruta[0] else "")

                # Mostrar el riesgo del tramo
                riesgo = riesgo_tramo.get((origen, destino), 5)
                mid_x = (coords[origen, 0] + coords[destino, 0]) / 2
                mid_y = (coords[origen, 1] + coords[destino, 1]) / 2
                plt.annotate(f"R:{riesgo}", (mid_x, mid_y), fontsize=9,
                             ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

        plt.title(titulo)
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')

        # Crear leyenda sin duplicados
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        plt.grid(True)
        plt.axis('equal')

        plt.savefig(
            f'img/punto_2/{titulo.replace(" ", "_").lower()}.png', dpi=300)
        plt.show()

        # Imprimir resumen de la solución
        print(f"\nResumen de la solución: {titulo}")
        print(f"Distancia total: {solucion['distancia']:.2f}")
        print(f"Calidad de inspección: {solucion['calidad']:.2f}")
        print(f"Nivel de riesgo: {solucion['riesgo']:.2f}")

        print("\nRutas por equipo:")
        for equipo, ruta in rutas.items():
            print(f"Equipo {equipo}: ", end="")
            ruta_str = "Depósito"
            for origen, destino in ruta:
                ruta_str += f" -> {destino if destino != 0 else 'Depósito'}"
            print(ruta_str)

            # Calcular métricas por equipo
            distancia_equipo = sum(
                distancias[origen, destino] for origen, destino in ruta)
            calidad_equipo = sum(calidad_inspeccion.get(destino, 0)
                                 for _, destino in ruta if destino != 0)
            riesgo_equipo = sum(riesgo_tramo.get((origen, destino), 5)
                                for origen, destino in ruta)

            print(f"  Distancia: {distancia_equipo:.2f}")
            print(f"  Calidad: {calidad_equipo:.2f}")
            print(f"  Riesgo: {riesgo_equipo:.2f}")
    except Exception as e:
        print(f"Error al visualizar solución: {e}")


def comparar_metodos(df_sp, df_ec):
    """
    Compara los resultados obtenidos por los diferentes métodos.

    Args:
        df_sp: DataFrame con resultados del método de suma ponderada
        df_ec: DataFrame con resultados del método epsilon-constraint
    """
    print("\nComparación de Métodos:")

    # Verificar si hay datos para comparar
    if len(df_sp) == 0 or len(df_ec) == 0:
        print("No hay suficientes datos para comparar los métodos.")
        return

    # Identificar soluciones Pareto-óptimas
    df_sp['es_pareto_optimo'] = es_pareto_optimo(df_sp)
    df_ec['es_pareto_optimo'] = es_pareto_optimo(df_ec)

    # Filtrar solo soluciones óptimas
    df_sp_pareto = df_sp[df_sp['es_pareto_optimo']].reset_index(drop=True)
    df_ec_pareto = df_ec[df_ec['es_pareto_optimo']].reset_index(drop=True)

    print(
        f"Método de Suma Ponderada: {len(df_sp)} soluciones generadas, {len(df_sp_pareto)} Pareto-óptimas")
    print(
        f"Método Epsilon-Constraint: {len(df_ec)} soluciones generadas, {len(df_ec_pareto)} Pareto-óptimas")

    # Calcular métricas para comparar los métodos
    # 1. Cobertura del espacio de soluciones
    if len(df_sp_pareto) > 0:
        sp_min_dist = df_sp_pareto['distancia'].min()
        sp_max_dist = df_sp_pareto['distancia'].max()
        sp_min_cal = df_sp_pareto['calidad'].min()
        sp_max_cal = df_sp_pareto['calidad'].max()
        sp_min_risk = df_sp_pareto['riesgo'].min()
        sp_max_risk = df_sp_pareto['riesgo'].max()
    else:
        sp_min_dist = sp_max_dist = sp_min_cal = sp_max_cal = sp_min_risk = sp_max_risk = 0

    if len(df_ec_pareto) > 0:
        ec_min_dist = df_ec_pareto['distancia'].min()
        ec_max_dist = df_ec_pareto['distancia'].max()
        ec_min_cal = df_ec_pareto['calidad'].min()
        ec_max_cal = df_ec_pareto['calidad'].max()
        ec_min_risk = df_ec_pareto['riesgo'].min()
        ec_max_risk = df_ec_pareto['riesgo'].max()
    else:
        ec_min_dist = ec_max_dist = ec_min_cal = ec_max_cal = ec_min_risk = ec_max_risk = 0

    print("\nCobertura del espacio de soluciones:")
    print("Método de Suma Ponderada:")
    print(f"  Rango de distancia: [{sp_min_dist:.2f}, {sp_max_dist:.2f}]")
    print(f"  Rango de calidad: [{sp_min_cal:.2f}, {sp_max_cal:.2f}]")
    print(f"  Rango de riesgo: [{sp_min_risk:.2f}, {sp_max_risk:.2f}]")

    print("Método Epsilon-Constraint:")
    print(f"  Rango de distancia: [{ec_min_dist:.2f}, {ec_max_dist:.2f}]")
    print(f"  Rango de calidad: [{ec_min_cal:.2f}, {ec_max_cal:.2f}]")
    print(f"  Rango de riesgo: [{ec_min_risk:.2f}, {ec_max_risk:.2f}]")

    # 2. Diversidad de soluciones
    def calcular_spacing(df):
        """Calcula el espaciado entre soluciones en el frente de Pareto."""
        if len(df) <= 1:
            return 0, 0

        try:
            # Normalizar objetivos para cálculo justo
            dist_min, dist_max = df['distancia'].min(), df['distancia'].max()
            cal_min, cal_max = df['calidad'].min(), df['calidad'].max()
            risk_min, risk_max = df['riesgo'].min(), df['riesgo'].max()

            # Evitar división por cero
            dist_range = max(dist_max - dist_min, 1e-10)
            cal_range = max(cal_max - cal_min, 1e-10)
            risk_range = max(risk_max - risk_min, 1e-10)

            dist_norm = (df['distancia'] - dist_min) / dist_range
            cal_norm = (df['calidad'] - cal_min) / cal_range
            risk_norm = (df['riesgo'] - risk_min) / risk_range

            # Calcular distancias entre cada par de soluciones
            distancias = []
            for i in range(len(df)):
                dist_min = float('inf')
                for j in range(len(df)):
                    if i != j:
                        d = np.sqrt((dist_norm.iloc[i] - dist_norm.iloc[j])**2 +
                                    (cal_norm.iloc[i] - cal_norm.iloc[j])**2 +
                                    (risk_norm.iloc[i] - risk_norm.iloc[j])**2)
                        dist_min = min(dist_min, d)
                if dist_min < float('inf'):
                    distancias.append(dist_min)

            # Calcular estadísticas de espaciado
            if len(distancias) > 0:
                d_mean = np.mean(distancias)
                spacing = np.sqrt(
                    np.sum((d_mean - np.array(distancias))**2) / len(distancias))
                return d_mean, spacing
            else:
                return 0, 0
        except Exception as e:
            print(f"Error al calcular spacing: {e}")
            return 0, 0

    sp_d_mean, sp_spacing = calcular_spacing(df_sp_pareto)
    ec_d_mean, ec_spacing = calcular_spacing(df_ec_pareto)

    print("\nDiversidad de soluciones:")
    print(
        f"Método de Suma Ponderada: Distancia media = {sp_d_mean:.4f}, Spacing = {sp_spacing:.4f}")
    print(
        f"Método Epsilon-Constraint: Distancia media = {ec_d_mean:.4f}, Spacing = {ec_spacing:.4f}")

    # Visualizar comparación
    try:
        # Verificar si hay variación en calidad
        calidad_constante_sp = df_sp_pareto['calidad'].std(
        ) < 1e-10 if len(df_sp_pareto) > 0 else True
        calidad_constante_ec = df_ec_pareto['calidad'].std(
        ) < 1e-10 if len(df_ec_pareto) > 0 else True

        if calidad_constante_sp and calidad_constante_ec:
            # Si la calidad es constante, hacemos una visualización 2D
            plt.figure(figsize=(12, 8))

            if len(df_sp_pareto) > 0:
                plt.scatter(df_sp_pareto['distancia'], df_sp_pareto['riesgo'],
                            c='blue', s=80, alpha=0.7, label='Suma Ponderada')

            if len(df_ec_pareto) > 0:
                plt.scatter(df_ec_pareto['distancia'], df_ec_pareto['riesgo'],
                            c='red', s=80, alpha=0.7, label='Epsilon-Constraint')

            plt.xlabel('Distancia Total')
            plt.ylabel('Riesgo Total')
            plt.title(
                'Comparación de Métodos: Soluciones Pareto-óptimas (Calidad constante)')

            plt.legend()
            plt.grid(True)
        else:
            # Visualización 3D normal
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            if len(df_sp_pareto) > 0:
                ax.scatter(df_sp_pareto['distancia'], df_sp_pareto['calidad'], df_sp_pareto['riesgo'],
                           c='blue', s=80, alpha=0.7, label='Suma Ponderada')

            if len(df_ec_pareto) > 0:
                ax.scatter(df_ec_pareto['distancia'], df_ec_pareto['calidad'], df_ec_pareto['riesgo'],
                           c='red', s=80, alpha=0.7, label='Epsilon-Constraint')

            ax.set_xlabel('Distancia Total')
            ax.set_ylabel('Calidad de Inspección')
            ax.set_zlabel('Riesgo Total')
            ax.set_title('Comparación de Métodos: Soluciones Pareto-óptimas')

            ax.legend()

        plt.savefig('img/punto_2/comparacion_metodos.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error al visualizar comparación: {e}")

    # Conclusión
    conclusion = ""
    if len(df_sp_pareto) > len(df_ec_pareto):
        conclusion += "El método de Suma Ponderada generó más soluciones Pareto-óptimas. "
    else:
        conclusion += "El método Epsilon-Constraint generó más soluciones Pareto-óptimas. "

    if sp_spacing < ec_spacing:
        conclusion += "El método de Suma Ponderada produjo soluciones con distribución más uniforme. "
    else:
        conclusion += "El método Epsilon-Constraint produjo soluciones con distribución más uniforme. "

    if ((sp_max_dist - sp_min_dist) > (ec_max_dist - ec_min_dist) or
        (sp_max_cal - sp_min_cal) > (ec_max_cal - ec_min_cal) or
            (sp_max_risk - sp_min_risk) > (ec_max_risk - ec_min_risk)):
        conclusion += "El método de Suma Ponderada cubrió un mayor rango del espacio de soluciones."
    else:
        conclusion += "El método Epsilon-Constraint cubrió un mayor rango del espacio de soluciones."

    print("\nConclusión:")
    print(conclusion)


def analisis_sensibilidad(df_pareto, valores_extremos):
    """
    Realiza un análisis de sensibilidad variando la importancia relativa de los objetivos.

    Args:
        df_pareto: DataFrame con soluciones Pareto-óptimas
        valores_extremos: diccionario con valores extremos para normalización
    """
    print("\nAnálisis de Sensibilidad - Importancia Relativa de Objetivos:")

    if len(df_pareto) == 0:
        print("No hay suficientes soluciones para realizar el análisis de sensibilidad.")
        return pd.DataFrame()

    # Normalizar objetivos
    df_norm = df_pareto.copy()

    min_distancia = valores_extremos['min_distancia']
    max_distancia = valores_extremos['max_distancia']
    min_calidad = valores_extremos['min_calidad']
    max_calidad = valores_extremos['max_calidad']
    min_riesgo = valores_extremos['min_riesgo']
    max_riesgo = valores_extremos['max_riesgo']

    # Evitar división por cero
    distancia_range = max(max_distancia - min_distancia, 1e-10)
    calidad_range = max(max_calidad - min_calidad, 1e-10)
    riesgo_range = max(max_riesgo - min_riesgo, 1e-10)

    df_norm['distancia_norm'] = (
        df_pareto['distancia'] - min_distancia) / distancia_range
    df_norm['calidad_norm'] = (
        df_pareto['calidad'] - min_calidad) / calidad_range
    df_norm['riesgo_norm'] = (df_pareto['riesgo'] - min_riesgo) / riesgo_range

    # Verificar si hay variación en la calidad
    calidad_constante = df_pareto['calidad'].std() < 1e-10

    # Escenarios de análisis de sensibilidad
    escenarios = [
        {"nombre": "Equilibrado", "pesos": [1/3, 1/3, 1/3]},
        {"nombre": "Prioridad Distancia", "pesos": [0.6, 0.2, 0.2]},
        {"nombre": "Prioridad Calidad", "pesos": [0.2, 0.6, 0.2]},
        {"nombre": "Prioridad Riesgo", "pesos": [0.2, 0.2, 0.6]},
        {"nombre": "Distancia Dominante", "pesos": [0.8, 0.1, 0.1]},
        {"nombre": "Calidad Dominante", "pesos": [0.1, 0.8, 0.1]},
        {"nombre": "Riesgo Dominante", "pesos": [0.1, 0.1, 0.8]}
    ]

    resultados_sensibilidad = []

    for escenario in escenarios:
        try:
            # Calcular puntaje ponderado para cada solución
            df_norm['score'] = escenario['pesos'][0] * (1 - df_norm['distancia_norm']) + \
                escenario['pesos'][1] * df_norm['calidad_norm'] + \
                escenario['pesos'][2] * (1 - df_norm['riesgo_norm'])

            # Encontrar la mejor solución para este escenario (manejar NaN)
            try:
                idx_mejor = df_norm['score'].idxmax()
                if pd.isna(idx_mejor):
                    idx_mejor = 0  # Valor predeterminado si es NaN
            except:
                idx_mejor = 0  # Valor predeterminado en caso de error

            resultados_sensibilidad.append({
                'escenario': escenario['nombre'],
                'peso_distancia': escenario['pesos'][0],
                'peso_calidad': escenario['pesos'][1],
                'peso_riesgo': escenario['pesos'][2],
                'idx_mejor': idx_mejor,
                'distancia': df_pareto.loc[idx_mejor, 'distancia'],
                'calidad': df_pareto.loc[idx_mejor, 'calidad'],
                'riesgo': df_pareto.loc[idx_mejor, 'riesgo'],
                'score': df_norm.loc[idx_mejor, 'score'] if not pd.isna(df_norm.loc[idx_mejor, 'score']) else 0
            })

            print(
                f"\nEscenario: {escenario['nombre']} (Pesos: {escenario['pesos']})")
            print(f"  Mejor solución: Índice {idx_mejor}")
            print(f"  Distancia: {df_pareto.loc[idx_mejor, 'distancia']:.2f}")
            print(f"  Calidad: {df_pareto.loc[idx_mejor, 'calidad']:.2f}")
            print(f"  Riesgo: {df_pareto.loc[idx_mejor, 'riesgo']:.2f}")

            score_value = df_norm.loc[idx_mejor, 'score']
            if not pd.isna(score_value):
                print(f"  Score ponderado: {score_value:.4f}")
            else:
                print("  Score ponderado: No disponible")
        except Exception as e:
            print(f"Error al procesar escenario {escenario['nombre']}: {e}")

    if len(resultados_sensibilidad) == 0:
        print("No se pudieron generar resultados para el análisis de sensibilidad.")
        return pd.DataFrame()

    df_sensibilidad = pd.DataFrame(resultados_sensibilidad)

    # Visualizar resultados de sensibilidad
    try:
        if calidad_constante:
            # Para un problema con calidad constante, visualizamos en 2D
            plt.figure(figsize=(12, 8))

            plt.scatter(df_pareto['distancia'], df_pareto['riesgo'],
                        c='lightgray', s=100, alpha=0.3, label='Todas las soluciones Pareto')

            # Colores para diferentes escenarios
            colores = ['blue', 'red', 'green',
                       'purple', 'orange', 'brown', 'pink']
            markers = ['o', 's', '^', 'D', '*', 'p', 'h']

            # Graficar las mejores soluciones para cada escenario
            for i, row in df_sensibilidad.iterrows():
                plt.scatter(row['distancia'], row['riesgo'],
                            c=colores[i % len(colores)], s=150, label=row['escenario'],
                            marker=markers[i % len(markers)])

            plt.xlabel('Distancia Total')
            plt.ylabel('Riesgo Total')
            plt.title(
                'Análisis de Sensibilidad - Importancia Relativa de Objetivos')

            # Añadir leyenda
            plt.legend()
        else:
            # Visualización 3D para problemas con variación en los tres objetivos
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(df_pareto['distancia'], df_pareto['calidad'], df_pareto['riesgo'],
                       c='lightgray', s=30, alpha=0.3, label='Todas las soluciones Pareto')

            # Colores para diferentes escenarios
            colores = ['blue', 'red', 'green',
                       'purple', 'orange', 'brown', 'pink']
            markers = ['o', 's', '^', 'D', '*', 'p', 'h']

            # Graficar las mejores soluciones para cada escenario
            for i, row in df_sensibilidad.iterrows():
                ax.scatter(row['distancia'], row['calidad'], row['riesgo'],
                           c=colores[i % len(colores)], s=150, label=row['escenario'],
                           marker=markers[i % len(markers)])

            ax.set_xlabel('Distancia Total')
            ax.set_ylabel('Calidad de Inspección')
            ax.set_zlabel('Riesgo Total')
            ax.set_title(
                'Análisis de Sensibilidad - Importancia Relativa de Objetivos')

            # Añadir leyenda
            ax.legend()

        plt.savefig('img/punto_2/analisis_sensibilidad.png', dpi=300)
        plt.show()

        # Comparativas para mejor visualización
        if calidad_constante:
            # Si la calidad es constante, solo comparamos distancia y riesgo
            fig, axs = plt.subplots(2, 1, figsize=(14, 12))

            # Variación en distancia por escenario
            axs[0].bar(df_sensibilidad['escenario'],
                       df_sensibilidad['distancia'], color='blue')
            axs[0].set_ylabel('Distancia Total')
            axs[0].set_title('Variación de Distancia por Escenario')
            axs[0].grid(axis='y')
            plt.setp(axs[0].xaxis.get_majorticklabels(),
                     rotation=45, ha='right')

            # Variación en riesgo por escenario
            axs[1].bar(df_sensibilidad['escenario'],
                       df_sensibilidad['riesgo'], color='red')
            axs[1].set_ylabel('Riesgo Total')
            axs[1].set_title('Variación de Riesgo por Escenario')
            axs[1].grid(axis='y')
            plt.setp(axs[1].xaxis.get_majorticklabels(),
                     rotation=45, ha='right')
        else:
            # Si hay variación en los tres objetivos, mostramos los tres
            fig, axs = plt.subplots(3, 1, figsize=(14, 18))

            # Variación en distancia por escenario
            axs[0].bar(df_sensibilidad['escenario'],
                       df_sensibilidad['distancia'], color='blue')
            axs[0].set_ylabel('Distancia Total')
            axs[0].set_title('Variación de Distancia por Escenario')
            axs[0].grid(axis='y')
            plt.setp(axs[0].xaxis.get_majorticklabels(),
                     rotation=45, ha='right')

            # Variación en calidad por escenario
            axs[1].bar(df_sensibilidad['escenario'],
                       df_sensibilidad['calidad'], color='green')
            axs[1].set_ylabel('Calidad de Inspección')
            axs[1].set_title('Variación de Calidad por Escenario')
            axs[1].grid(axis='y')
            plt.setp(axs[1].xaxis.get_majorticklabels(),
                     rotation=45, ha='right')

            # Variación en riesgo por escenario
            axs[2].bar(df_sensibilidad['escenario'],
                       df_sensibilidad['riesgo'], color='red')
            axs[2].set_ylabel('Riesgo Total')
            axs[2].set_title('Variación de Riesgo por Escenario')
            axs[2].grid(axis='y')
            plt.setp(axs[2].xaxis.get_majorticklabels(),
                     rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('img/punto_2/comparativa_escenarios.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error al visualizar resultados: {e}")

    return df_sensibilidad


def principal():
    """Función principal que ejecuta el flujo completo de análisis."""
    print("=" * 80)
    print("PROBLEMA 2: OPTIMIZACIÓN MULTIOBJETIVO EN PLANIFICACIÓN DE RUTAS DE INSPECCIÓN")
    print("=" * 80)

    try:
        # Crear carpeta para guardar imágenes si no existe
        if not os.path.exists('img'):
            os.makedirs('img')

        # 1. Calcular valores extremos
        print("\n1. CALCULANDO VALORES EXTREMOS PARA NORMALIZACIÓN...")
        valores_extremos = calcular_valores_extremos()

        # 2. Aplicar método de la suma ponderada
        print("\n2. APLICANDO MÉTODO DE LA SUMA PONDERADA...")
        resultados_sp = metodo_suma_ponderada(
            num_puntos=4, valores_extremos=valores_extremos)

        # Identificar soluciones Pareto-óptimas
        resultados_sp['es_pareto_optimo'] = es_pareto_optimo(resultados_sp)
        df_sp_pareto = resultados_sp[resultados_sp['es_pareto_optimo']].reset_index(
            drop=True)

        # Visualizar frente de Pareto
        graficar_frente_pareto_3d(resultados_sp, "Suma Ponderada")

        # 3. Aplicar método ε-constraint
        print("\n3. APLICANDO MÉTODO ε-CONSTRAINT...")
        resultados_ec = metodo_epsilon_constraint(
            num_puntos=3, valores_extremos=valores_extremos)

        # Identificar soluciones Pareto-óptimas
        resultados_ec['es_pareto_optimo'] = es_pareto_optimo(resultados_ec)
        df_ec_pareto = resultados_ec[resultados_ec['es_pareto_optimo']].reset_index(
            drop=True)

        # Visualizar frente de Pareto
        graficar_frente_pareto_3d(resultados_ec, "Epsilon-Constraint")

        # 4. Comparar métodos
        print("\n4. COMPARANDO MÉTODOS...")
        comparar_metodos(resultados_sp, resultados_ec)

        # Usamos las soluciones Pareto-óptimas del método que generó más
        if len(df_sp_pareto) >= len(df_ec_pareto):
            df_pareto = df_sp_pareto
            print(
                "\nUsando soluciones Pareto-óptimas del método de la Suma Ponderada para análisis adicional.")
        else:
            df_pareto = df_ec_pareto
            print(
                "\nUsando soluciones Pareto-óptimas del método Epsilon-Constraint para análisis adicional.")

        # 5. Analizar trade-offs
        try:
            print("\n5. ANALIZANDO TRADE-OFFS ENTRE OBJETIVOS...")
            analizar_tradeoffs(df_pareto)
        except Exception as e:
            print(f"\nError en el análisis de trade-offs: {e}")

        # 6. Análisis de sensibilidad
        try:
            print("\n6. REALIZANDO ANÁLISIS DE SENSIBILIDAD...")
            df_sensibilidad = analisis_sensibilidad(
                df_pareto, valores_extremos)
        except Exception as e:
            print(f"\nError en el análisis de sensibilidad: {e}")
            df_sensibilidad = pd.DataFrame()

        # 7. Identificar mejor solución de compromiso
        try:
            print("\n7. IDENTIFICANDO LA MEJOR SOLUCIÓN DE COMPROMISO...")
            mejor_indice = identificar_mejor_solucion_compromiso(
                df_pareto, valores_extremos)
        except Exception as e:
            print(f"\nError al identificar la mejor solución: {e}")
            mejor_indice = 0 if len(df_pareto) > 0 else None

        # 8. Visualizar las soluciones
        try:
            if mejor_indice is not None and len(df_pareto) > 0:
                print("\n8. VISUALIZANDO LA MEJOR SOLUCIÓN DE COMPROMISO...")
                visualizar_solucion(
                    df_pareto.iloc[mejor_indice], "Mejor Solución de Compromiso")

                # 9. Visualizar soluciones extremas para comparación
                print("\n9. VISUALIZANDO SOLUCIONES EXTREMAS PARA COMPARACIÓN...")

                # Solución con mínima distancia
                idx_min_dist = df_pareto['distancia'].idxmin()
                visualizar_solucion(
                    df_pareto.iloc[idx_min_dist], "Solución con Mínima Distancia")

                # Solución con mínimo riesgo
                idx_min_risk = df_pareto['riesgo'].idxmin()
                visualizar_solucion(
                    df_pareto.iloc[idx_min_risk], "Solución con Mínimo Riesgo")
            else:
                print("\nNo hay soluciones disponibles para visualizar.")
        except Exception as e:
            print(f"\nError al visualizar soluciones: {e}")

        print("\nANÁLISIS COMPLETO FINALIZADO")
        print("Los resultados han sido guardados en archivos de imagen en la carpeta 'img' para su inclusión en el informe.")

    except Exception as e:
        print(f"\nError durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    principal()
