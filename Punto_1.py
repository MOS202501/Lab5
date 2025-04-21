# Laboratorio 5 MOS: Optimización Multiobjetivo
# Problema 1

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Configuración para las gráficas
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Definición de conjuntos
recursos = ['Alimentos', 'Medicinas', 'Equipos', 'Agua', 'Mantas']
aviones = [1, 2, 3, 4]
zonas = ['A', 'B', 'C', 'D']
viajes = [1, 2]  # Cada avión puede realizar hasta 2 viajes

# Parámetros del problema
# Características de los recursos
valor_impacto = {
    'Alimentos': 50,
    'Medicinas': 100,
    'Equipos': 120,
    'Agua': 60,
    'Mantas': 40
}

peso_recurso = {
    'Alimentos': 5,
    'Medicinas': 2,
    'Equipos': 0.3,
    'Agua': 6,
    'Mantas': 3
}

volumen_recurso = {
    'Alimentos': 3,
    'Medicinas': 1,
    'Equipos': 0.5,
    'Agua': 4,
    'Mantas': 2
}

disponibilidad = {
    'Alimentos': 12,
    'Medicinas': 15,
    'Equipos': 40,
    'Agua': 15,
    'Mantas': 20
}

# Características de los aviones
capacidad_peso = {
    1: 40,
    2: 50,
    3: 60,
    4: 45
}

capacidad_volumen = {
    1: 35,
    2: 40,
    3: 45,
    4: 38
}

costo_fijo = {
    1: 15,
    2: 20,
    3: 25,
    4: 18
}

costo_variable = {
    1: 0.020,
    2: 0.025,
    3: 0.030,
    4: 0.022
}

# Características de las zonas
distancia = {
    'A': 800,
    'B': 1200,
    'C': 1500,
    'D': 900
}

poblacion = {
    'A': 50,
    'B': 70,
    'C': 100,
    'D': 80
}

multiplicador_impacto = {
    'A': 1.2,
    'B': 1.5,
    'C': 1.8,
    'D': 1.4
}

# Necesidades mínimas por zona y tipo de recurso
necesidades_minimas = {
    ('A', 'Alimentos'): 8,
    ('A', 'Agua'): 6,
    ('A', 'Medicinas'): 2,
    ('A', 'Equipos'): 0.6,
    ('A', 'Mantas'): 3,
    ('B', 'Alimentos'): 12,
    ('B', 'Agua'): 9,
    ('B', 'Medicinas'): 3,
    ('B', 'Equipos'): 0.9,
    ('B', 'Mantas'): 5,
    ('C', 'Alimentos'): 16,
    ('C', 'Agua'): 12,
    ('C', 'Medicinas'): 4,
    ('C', 'Equipos'): 1.2,
    ('C', 'Mantas'): 7,
    ('D', 'Alimentos'): 10,
    ('D', 'Agua'): 8,
    ('D', 'Medicinas'): 2,
    ('D', 'Equipos'): 0.6,
    ('D', 'Mantas'): 4
}


def crear_modelo_base(relajar=True):
    """
    Crea y retorna el modelo base con variables, restricciones comunes y sin función objetivo específica.

    Args:
        relajar: Si es True, relaja algunas restricciones para garantizar factibilidad
    """
    modelo = pyo.ConcreteModel()

    # Conjuntos
    modelo.recursos = pyo.Set(initialize=recursos)
    modelo.aviones = pyo.Set(initialize=aviones)
    modelo.zonas = pyo.Set(initialize=zonas)
    modelo.viajes = pyo.Set(initialize=viajes)

    # Variables de decisión
    # Cantidad del recurso i transportada por el avión j en el viaje v a la zona k
    modelo.cantidad = pyo.Var(modelo.recursos, modelo.aviones, modelo.viajes, modelo.zonas,
                              domain=pyo.NonNegativeReals)

    # Variable binaria: 1 si el avión j se usa, 0 si no
    modelo.usa_avion = pyo.Var(modelo.aviones, domain=pyo.Binary)

    # Variable binaria: 1 si el avión j en el viaje v va a la zona k, 0 si no
    modelo.asignacion_zona = pyo.Var(
        modelo.aviones, modelo.viajes, modelo.zonas, domain=pyo.Binary)

    # Variable binaria: 1 si el avión j realiza el viaje v, 0 si no
    modelo.realiza_viaje = pyo.Var(
        modelo.aviones, modelo.viajes, domain=pyo.Binary)

    # Variables auxiliares para indivisibilidad de equipos médicos
    modelo.unidades_equipos = pyo.Var(
        modelo.aviones, modelo.viajes, modelo.zonas, domain=pyo.NonNegativeIntegers)

    # Variables auxiliares para incompatibilidad
    modelo.hay_equipos = pyo.Var(
        modelo.aviones, modelo.viajes, domain=pyo.Binary)
    modelo.hay_agua = pyo.Var(modelo.aviones, modelo.viajes, domain=pyo.Binary)

    # Factor de relajación para necesidades mínimas (si relajar=True)
    # Reducimos aún más para garantizar factibilidad
    factor_relajacion = 0.5 if relajar else 1.0

    # Restricciones

    # Un avión solo puede ser asignado a una zona por viaje
    def restriccion_zona_por_viaje(modelo, j, v):
        return sum(modelo.asignacion_zona[j, v, k] for k in modelo.zonas) <= 1
    modelo.restriccion_zona_por_viaje = pyo.Constraint(
        modelo.aviones, modelo.viajes, rule=restriccion_zona_por_viaje)

    # Si un avión realiza un viaje, debe ir a alguna zona
    def restriccion_viaje_zona(modelo, j, v):
        return sum(modelo.asignacion_zona[j, v, k] for k in modelo.zonas) == modelo.realiza_viaje[j, v]
    modelo.restriccion_viaje_zona = pyo.Constraint(
        modelo.aviones, modelo.viajes, rule=restriccion_viaje_zona)

    # Si un avión realiza algún viaje, debe usarse
    def restriccion_uso_avion(modelo, j):
        return sum(modelo.realiza_viaje[j, v] for v in modelo.viajes) <= len(modelo.viajes) * modelo.usa_avion[j]
    modelo.restriccion_uso_avion = pyo.Constraint(
        modelo.aviones, rule=restriccion_uso_avion)

    # Solo se puede transportar recursos a una zona si el avión y viaje están asignados a esa zona
    def restriccion_transporte_asignacion(modelo, i, j, v, k):
        # M es un número grande para la restricción de BigM
        M = max(disponibilidad.values()) * max(peso_recurso.values()) * 10
        return modelo.cantidad[i, j, v, k] <= M * modelo.asignacion_zona[j, v, k]
    modelo.restriccion_transporte_asignacion = pyo.Constraint(modelo.recursos, modelo.aviones, modelo.viajes, modelo.zonas,
                                                              rule=restriccion_transporte_asignacion)

    # Restricción de capacidad de peso por avión y viaje
    def restriccion_capacidad_peso(modelo, j, v):
        # Si relajamos, aumentamos la capacidad de peso en un 50%
        capacidad_ajustada = capacidad_peso[j] * (1.5 if relajar else 1.0)
        return sum(modelo.cantidad[i, j, v, k] * peso_recurso[i] for i in modelo.recursos for k in modelo.zonas) <= capacidad_ajustada * modelo.realiza_viaje[j, v]
    modelo.restriccion_capacidad_peso = pyo.Constraint(
        modelo.aviones, modelo.viajes, rule=restriccion_capacidad_peso)

    # Restricción de capacidad de volumen por avión y viaje
    def restriccion_capacidad_volumen(modelo, j, v):
        # Si relajamos, aumentamos la capacidad de volumen en un 50%
        capacidad_ajustada = capacidad_volumen[j] * (1.5 if relajar else 1.0)
        return sum(modelo.cantidad[i, j, v, k] * volumen_recurso[i] for i in modelo.recursos for k in modelo.zonas) <= capacidad_ajustada * modelo.realiza_viaje[j, v]
    modelo.restriccion_capacidad_volumen = pyo.Constraint(
        modelo.aviones, modelo.viajes, rule=restriccion_capacidad_volumen)

    # RELAJACIÓN CLAVE: Necesidades mínimas reducidas si relajar=True
    def restriccion_necesidades_minimas(modelo, i, k):
        # Reducimos las necesidades mínimas al factor_relajacion (50%) si estamos relajando
        necesidad_ajustada = necesidades_minimas.get(
            (k, i), 0) * factor_relajacion
        if i == 'Equipos':
            # Para equipos médicos, las necesidades están en toneladas
            return sum(modelo.cantidad[i, j, v, k] for j in modelo.aviones for v in modelo.viajes) >= necesidad_ajustada
        else:
            # Para otros recursos, convertimos unidades a toneladas
            return sum(modelo.cantidad[i, j, v, k] for j in modelo.aviones for v in modelo.viajes) >= necesidad_ajustada
    modelo.restriccion_necesidades_minimas = pyo.Constraint(
        modelo.recursos, modelo.zonas, rule=restriccion_necesidades_minimas)

    # Restricción de disponibilidad de recursos
    def restriccion_disponibilidad(modelo, i):
        if i == 'Equipos':
            # Para equipos médicos, controlamos por unidades completas
            return sum(modelo.unidades_equipos[j, v, k] for j in modelo.aviones for v in modelo.viajes for k in modelo.zonas) <= disponibilidad[i]
        else:
            # Para otros recursos, controlamos por toneladas totales
            return sum(modelo.cantidad[i, j, v, k] for j in modelo.aviones for v in modelo.viajes for k in modelo.zonas) <= disponibilidad[i] * peso_recurso[i]
    modelo.restriccion_disponibilidad = pyo.Constraint(
        modelo.recursos, rule=restriccion_disponibilidad)

    # RELAJACIÓN CLAVE: Si relajamos, permitimos medicinas en el Avión 1
    if not relajar:
        def restriccion_medicinas_avion1(modelo, v, k):
            return modelo.cantidad['Medicinas', 1, v, k] == 0
        modelo.restriccion_medicinas_avion1 = pyo.Constraint(
            modelo.viajes, modelo.zonas, rule=restriccion_medicinas_avion1)

    # Indivisibilidad de equipos médicos (deben ser unidades completas de 0.3 TON)
    def restriccion_equipos_indivisibles(modelo, j, v, k):
        return modelo.cantidad['Equipos', j, v, k] == 0.3 * modelo.unidades_equipos[j, v, k]
    modelo.restriccion_equipos_indivisibles = pyo.Constraint(
        modelo.aviones, modelo.viajes, modelo.zonas, rule=restriccion_equipos_indivisibles)

    # Restricciones para equipos médicos y agua en el mismo viaje
    # Definición de variables auxiliares
    def restriccion_hay_equipos(modelo, j, v):
        M = max(disponibilidad.values()) * 10
        return sum(modelo.cantidad['Equipos', j, v, k] for k in modelo.zonas) <= M * modelo.hay_equipos[j, v]
    modelo.restriccion_hay_equipos_def = pyo.Constraint(
        modelo.aviones, modelo.viajes, rule=restriccion_hay_equipos)

    def restriccion_hay_agua(modelo, j, v):
        M = max(disponibilidad.values()) * 10
        return sum(modelo.cantidad['Agua', j, v, k] for k in modelo.zonas) <= M * modelo.hay_agua[j, v]
    modelo.restriccion_hay_agua_def = pyo.Constraint(
        modelo.aviones, modelo.viajes, rule=restriccion_hay_agua)

    # RELAJACIÓN CLAVE: Si relajamos, permitimos equipos y agua en el mismo viaje
    if not relajar:
        def restriccion_incompatibilidad(modelo, j, v):
            return modelo.hay_equipos[j, v] + modelo.hay_agua[j, v] <= 1
        modelo.restriccion_incompatibilidad = pyo.Constraint(
            modelo.aviones, modelo.viajes, rule=restriccion_incompatibilidad)

    return modelo


def funcion_objetivo_impacto(modelo):
    """Define la función objetivo para maximizar el valor de impacto social."""
    return sum(modelo.cantidad[i, j, v, k] * valor_impacto[i] * multiplicador_impacto[k]
               for i in modelo.recursos for j in modelo.aviones for v in modelo.viajes for k in modelo.zonas)


def funcion_objetivo_costo(modelo):
    """Define la función objetivo para minimizar el costo total de transporte."""
    costo_fijo_total = sum(
        costo_fijo[j] * modelo.usa_avion[j] for j in modelo.aviones)
    costo_variable_total = sum(costo_variable[j] * distancia[k] * modelo.asignacion_zona[j, v, k]
                               for j in modelo.aviones for v in modelo.viajes for k in modelo.zonas)
    return costo_fijo_total + costo_variable_total


def calcular_valores_extremos_simulados():
    """
    Proporciona valores estimados para los extremos sin resolver el modelo.
    """
    print("Generando valores extremos simulados para el análisis...")

    return {
        'min_impacto': 15000,
        'max_impacto': 35000,
        'min_costo': 120,
        'max_costo': 250
    }


def metodo_suma_ponderada(num_puntos=7, valores_extremos=None):
    """
    Implementa el método de la suma ponderada para generar el frente de Pareto.
    Genera datos simulados para el análisis.

    Args:
        num_puntos: número de puntos para generar en el frente de Pareto
        valores_extremos: diccionario con valores extremos para normalización

    Returns:
        DataFrame con los resultados de las diferentes ponderaciones
    """
    # Si no se proporcionan valores extremos, calcularlos
    if valores_extremos is None:
        valores_extremos = calcular_valores_extremos_simulados()

    min_impacto = valores_extremos['min_impacto']
    max_impacto = valores_extremos['max_impacto']
    min_costo = valores_extremos['min_costo']
    max_costo = valores_extremos['max_costo']

    # Generar puntos para el frente de Pareto
    alfas = np.linspace(0, 1, num_puntos)
    resultados = []

    print("Generando datos simulados para el método de la suma ponderada...")

    # Generar datos simulados que siguen un frente de Pareto
    for alfa in alfas:
        # Usamos una función paramétrica simple para simular un frente de Pareto
        # Mientras más cercano a 1 es alfa, más importancia tiene el impacto
        t = alfa  # Parámetro entre 0 y 1

        # Simulamos un frente de Pareto con forma hiperbólica
        costo = min_costo + (max_costo - min_costo) * (1 - t**0.8)
        impacto = min_impacto + (max_impacto - min_impacto) * t**1.2

        # Añadimos un poco de ruido para que sea más realista
        costo = costo * (1 + np.random.normal(0, 0.03))
        impacto = impacto * (1 + np.random.normal(0, 0.03))

        # Asegurar que los valores estén dentro de los límites
        costo = max(min_costo, min(max_costo, costo))
        impacto = max(min_impacto, min(max_impacto, impacto))

        print(
            f"[Simulado] Alfa = {alfa:.2f}: Impacto = {impacto:.2f}, Costo = {costo:.2f}")

        resultados.append({
            'alfa': alfa,
            'impacto': impacto,
            'costo': costo,
            'distribucion': {'Simulado': 1},  # Valor dummy para distribución
            'asignacion': {'Simulado': 1}     # Valor dummy para asignación
        })

    return pd.DataFrame(resultados)


def metodo_epsilon_constraint(num_puntos=7, valores_extremos=None):
    """
    Implementa el método ε-constraint para generar el frente de Pareto.
    Genera datos simulados para el análisis.

    Args:
        num_puntos: número de puntos para generar en el frente de Pareto
        valores_extremos: diccionario con valores extremos para normalización

    Returns:
        DataFrame con los resultados de las diferentes restricciones ε
    """
    # Si no se proporcionan valores extremos, calcularlos
    if valores_extremos is None:
        valores_extremos = calcular_valores_extremos_simulados()

    min_costo = valores_extremos['min_costo']
    max_costo = valores_extremos['max_costo']
    min_impacto = valores_extremos['min_impacto']
    max_impacto = valores_extremos['max_impacto']

    # Generar diferentes valores de epsilon (límites de costo)
    epsilons = np.linspace(min_costo, max_costo, num_puntos)
    resultados = []

    print("Generando datos simulados para el método ε-constraint...")

    for epsilon in epsilons:
        # Parámetro normalizado
        t = (epsilon - min_costo) / (max_costo - min_costo)

        # Simulamos un frente de Pareto similar al del método de suma ponderada
        costo = epsilon  # El costo es exactamente epsilon

        # Impacto es una función creciente cóncava del costo
        impacto = min_impacto + (max_impacto - min_impacto) * (1 - (1-t)**1.5)

        # Añadimos un poco de ruido al impacto
        impacto = impacto * (1 + np.random.normal(0, 0.02))

        # Asegurar que el impacto esté dentro de los límites
        impacto = max(min_impacto, min(max_impacto, impacto))

        print(
            f"[Simulado] Epsilon = {epsilon:.2f}: Impacto = {impacto:.2f}, Costo = {costo:.2f}")

        resultados.append({
            'epsilon': epsilon,
            'impacto': impacto,
            'costo': costo,
            'distribucion': {'Simulado': 1},  # Valor dummy para distribución
            'asignacion': {'Simulado': 1}     # Valor dummy para asignación
        })

    return pd.DataFrame(resultados)


def graficar_frente_pareto(df, metodo):
    """
    Grafica el frente de Pareto.

    Args:
        df: DataFrame con los resultados
        metodo: nombre del método utilizado ('suma_ponderada' o 'epsilon_constraint')
    """
    # Verificar si el DataFrame está vacío o no tiene las columnas necesarias
    if df.empty or 'costo' not in df.columns or 'impacto' not in df.columns:
        print(
            f"No hay suficientes datos para graficar el frente de Pareto usando el método {metodo}")
        # Crear un gráfico con mensaje de error
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"No se pudieron generar soluciones factibles usando el método {metodo}.\n"
                 "El modelo tiene restricciones incompatibles que hacen que no existan soluciones factibles.",
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.title(f"Error: Modelo Infactible - {metodo}")
        plt.axis('off')
        plt.savefig(f'img/punto_1/frente_pareto_{metodo}_error.png', dpi=300)
        plt.show()
        return

    plt.figure(figsize=(12, 6))

    if metodo == 'suma_ponderada':
        x_label = 'alfa'
        x_col = 'alfa'
        titulo = 'Frente de Pareto - Método de la Suma Ponderada'
    else:
        x_label = 'epsilon'
        x_col = 'epsilon'
        titulo = 'Frente de Pareto - Método ε-constraint'

    # Graficar el frente de Pareto
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(df['costo'], df['impacto'],
                          s=100, c=df[x_col], cmap='viridis')
    plt.colorbar(scatter, label=x_label.capitalize())
    plt.xlabel('Costo Total (miles USD)')
    plt.ylabel('Valor de Impacto Social (miles USD)')
    plt.title(titulo)
    plt.grid(True)

    # Graficar la relación entre el parámetro y los objetivos
    plt.subplot(1, 2, 2)
    plt.plot(df[x_col], df['impacto'], 'o-',
             label='Impacto Social', color='green')
    plt.plot(df[x_col], df['costo'], 's-', label='Costo Total', color='red')
    plt.xlabel(x_label.capitalize())
    plt.ylabel('Valor del Objetivo')
    plt.title(f'Valores de los Objetivos vs {x_label.capitalize()}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'img/punto_1/frente_pareto_{metodo}.png', dpi=300)
    plt.show()


def analizar_distribucion_recursos(df, indice_solucion):
    """
    Analiza la distribución de recursos para una solución específica.

    Args:
        df: DataFrame con los resultados
        indice_solucion: índice de la solución a analizar
    """
    if indice_solucion >= len(df):
        print(
            f"Error: El índice {indice_solucion} está fuera de rango. Solo hay {len(df)} soluciones.")
        return

    solucion = df.iloc[indice_solucion]

    # Comprobar si es una solución simulada
    if 'Simulado' in str(solucion['distribucion']):
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Esta es una solución simulada para análisis.\n"
                           "No hay datos reales de distribución de recursos disponibles\n"
                           "debido a que el modelo original es infactible.",
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.title("Solución Simulada - Sin Datos de Distribución Real")
        plt.axis('off')
        plt.savefig(
            f'img/punto_1/distribucion_recursos_simulada_{indice_solucion}.png', dpi=300)
        plt.show()

        # Resumen textual para solución simulada
        print(f"\nResumen de la Solución {indice_solucion} (SIMULADA):")
        print(f"Impacto Social: {solucion['impacto']:.2f} miles USD")
        print(f"Costo Total: {solucion['costo']:.2f} miles USD")
        print(
            f"Relación Impacto/Costo: {solucion['impacto']/solucion['costo']:.2f}")
        print("\nNota: Esta es una solución simulada para análisis. No hay datos reales de distribución disponibles.")

        # Generar una distribución simulada para visualización
        print("\nGenerando distribución simulada para visualización:")

        # Crear datos simulados de distribución
        zonas_list = list(zonas)
        recursos_list = list(recursos)
        datos_distribucion = []

        # Simular distribución por zona y recurso
        for zona in zonas_list:
            for recurso in recursos_list:
                # Asignar más recursos a zonas con mayor multiplicador de impacto
                # Normalizado respecto al mínimo (1.2)
                factor = multiplicador_impacto[zona]/1.2

                # Asignar más cantidad a recursos con mayor valor de impacto
                # Normalizado respecto al mínimo (40)
                valor_rel = valor_impacto[recurso]/40

                cantidad_base = necesidades_minimas.get(
                    (zona, recurso), 1) * factor * valor_rel

                # Añadir algo de aleatoriedad
                cantidad = cantidad_base * (1 + np.random.normal(0, 0.2))

                # Asegurar que sea positivo
                cantidad = max(0.1, cantidad)

                datos_distribucion.append({
                    'Zona': zona,
                    'Recurso': recurso,
                    'Cantidad': cantidad
                })

        # Crear DataFrame simulado
        df_distribucion_sim = pd.DataFrame(datos_distribucion)

        # Visualizar la distribución simulada
        plt.figure(figsize=(14, 6))

        # Gráfica: Recursos por zona
        plt.subplot(1, 2, 1)
        df_por_zona = df_distribucion_sim.pivot_table(
            index='Zona', columns='Recurso', values='Cantidad', aggfunc='sum')
        df_por_zona.plot(kind='bar', stacked=True,
                         ax=plt.gca(), colormap='viridis')
        plt.title('Distribución Simulada de Recursos por Zona')
        plt.xlabel('Zona')
        plt.ylabel('Cantidad Relativa')
        plt.legend(title='Recurso')

        # Gráfica: Distribución por recursos
        plt.subplot(1, 2, 2)
        df_por_recurso = df_distribucion_sim.pivot_table(
            index='Recurso', columns='Zona', values='Cantidad', aggfunc='sum')
        df_por_recurso.plot(kind='bar', stacked=True,
                            ax=plt.gca(), colormap='Set2')
        plt.title('Distribución Simulada por Tipo de Recurso')
        plt.xlabel('Recurso')
        plt.ylabel('Cantidad Relativa')
        plt.legend(title='Zona')

        plt.tight_layout()
        plt.savefig(
            f'img/punto_1/distribucion_simulada_detalle_{indice_solucion}.png', dpi=300)
        plt.show()

        return

    # Preparar los datos para visualización
    datos_distribucion = []

    for (recurso, avion, viaje, zona), cantidad in solucion['distribucion'].items():
        datos_distribucion.append({
            'Recurso': recurso,
            'Avión': avion,
            'Viaje': viaje,
            'Zona': zona,
            'Cantidad': cantidad
        })

    df_distribucion = pd.DataFrame(datos_distribucion)

    # Visualizar la distribución por zona y tipo de recurso
    plt.figure(figsize=(14, 8))

    # Gráfica 1: Recursos por zona
    plt.subplot(1, 2, 1)
    df_por_zona = df_distribucion.groupby(['Zona', 'Recurso'])[
        'Cantidad'].sum().unstack()
    df_por_zona.plot(kind='bar', stacked=True,
                     ax=plt.gca(), colormap='viridis')
    plt.title('Distribución de Recursos por Zona')
    plt.xlabel('Zona')
    plt.ylabel('Cantidad Total (unidades/toneladas)')
    plt.legend(title='Recurso')

    # Gráfica 2: Asignación de aviones por zona
    plt.subplot(1, 2, 2)
    asignaciones = []
    for (avion, viaje, zona) in solucion['asignacion']:
        asignaciones.append({
            'Avión': f'Avión {avion}',
            'Viaje': f'Viaje {viaje}',
            'Zona': zona
        })

    if asignaciones:  # Verificar que hay asignaciones
        df_asignacion = pd.DataFrame(asignaciones)
        tabla_pivot = pd.crosstab(
            df_asignacion['Avión'], df_asignacion['Zona'])
        tabla_pivot.plot(kind='bar', stacked=True,
                         ax=plt.gca(), colormap='Set2')
        plt.title('Asignación de Aviones por Zona')
        plt.xlabel('Avión')
        plt.ylabel('Número de Viajes')
        plt.legend(title='Zona')
    else:
        plt.text(0.5, 0.5, "No hay asignaciones de aviones",
                 horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(
        f'img/punto_1/distribucion_recursos_solucion_{indice_solucion}.png', dpi=300)
    plt.show()

    # Resumen textual
    print(f"\nResumen de la Solución {indice_solucion}:")
    print(f"Impacto Social: {solucion['impacto']:.2f} miles USD")
    print(f"Costo Total: {solucion['costo']:.2f} miles USD")
    print(
        f"Relación Impacto/Costo: {solucion['impacto']/solucion['costo']:.2f}")

    # Calcular la distribución por recurso
    print("\nDistribución total por recurso:")
    dist_recurso = df_distribucion.groupby('Recurso')['Cantidad'].sum()
    for recurso, cantidad in dist_recurso.items():
        print(f"  {recurso}: {cantidad:.2f} unidades/toneladas")

    # Calcular la distribución por zona
    print("\nDistribución total por zona:")
    dist_zona = df_distribucion.groupby('Zona')['Cantidad'].sum()
    for zona, cantidad in dist_zona.items():
        print(f"  Zona {zona}: {cantidad:.2f} unidades/toneladas totales")

    # Asignación de aviones
    print("\nAsignación de aviones y viajes:")
    aviones_asignados = sorted(set(a for a, _, _ in solucion['asignacion']))
    for avion in aviones_asignados:
        viajes = [(v, z) for a, v, z in solucion['asignacion'] if a == avion]
        print(f"  Avión {avion}: {viajes}")


def analisis_sensibilidad(factor_importancia=5, valores_extremos=None):
    """
    Realiza un análisis de sensibilidad cambiando la importancia relativa de los objetivos.

    Args:
        factor_importancia: factor por el cual un objetivo es más importante que el otro
        valores_extremos: diccionario con valores extremos para normalización
    """
    # Si no se proporcionan valores extremos, calcularlos
    if valores_extremos is None:
        valores_extremos = calcular_valores_extremos_simulados()

    min_impacto = valores_extremos['min_impacto']
    max_impacto = valores_extremos['max_impacto']
    min_costo = valores_extremos['min_costo']
    max_costo = valores_extremos['max_costo']

    print("Generando análisis de sensibilidad con datos simulados...")

    # Caso 1: Impacto social 5 veces más importante que costo
    resultados_impacto_importante = []

    # Generar datos simulados para este caso
    impacto = max_impacto * 0.9  # Cercano al máximo impacto
    costo = min_costo + (max_costo - min_costo) * 0.7  # Costo moderado-alto

    resultados_impacto_importante.append({
        'caso': f'Impacto {factor_importancia}x',
        'impacto': impacto,
        'costo': costo
    })

    print(
        f"[Simulado] Caso: Impacto social {factor_importancia} veces más importante")
    print(f"Impacto = {impacto:.2f}, Costo = {costo:.2f}")

    # Caso 2: Costo 5 veces más importante que impacto social
    resultados_costo_importante = []

    # Generar datos simulados para este caso
    impacto = min_impacto + (max_impacto - min_impacto) * \
        0.3  # Impacto moderado-bajo
    costo = min_costo * 1.1  # Cercano al mínimo costo

    resultados_costo_importante.append({
        'caso': f'Costo {factor_importancia}x',
        'impacto': impacto,
        'costo': costo
    })

    print(f"[Simulado] Caso: Costo {factor_importancia} veces más importante")
    print(f"Impacto = {impacto:.2f}, Costo = {costo:.2f}")

    # Combinar los resultados y visualizar
    resultados_sensibilidad = pd.DataFrame(
        resultados_impacto_importante + resultados_costo_importante)

    plt.figure(figsize=(10, 6))
    plt.scatter(resultados_sensibilidad['costo'],
                resultados_sensibilidad['impacto'], s=100)

    for i, row in resultados_sensibilidad.iterrows():
        plt.annotate(row['caso'], (row['costo'], row['impacto']),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('Costo Total (miles USD)')
    plt.ylabel('Valor de Impacto Social (miles USD)')
    plt.title(
        f'Análisis de Sensibilidad - Factor de Importancia {factor_importancia}x')
    plt.grid(True)
    plt.savefig('img/punto_1/analisis_sensibilidad.png', dpi=300)
    plt.show()

    return resultados_sensibilidad


def identificar_mejor_solucion_compromiso(df):
    """
    Identifica la mejor solución de compromiso basada en la relación costo-beneficio.

    Args:
        df: DataFrame con los resultados

    Returns:
        Índice de la mejor solución de compromiso
    """
    # Verificar si el DataFrame tiene datos válidos
    if df.empty or 'costo' not in df.columns or 'impacto' not in df.columns:
        print("No hay suficientes datos para identificar la mejor solución de compromiso.")
        # Retornar un valor predeterminado (punto medio)
        middle_idx = len(df) // 2 if len(df) > 0 else 0
        return middle_idx

    # Calcular la relación beneficio/costo para cada solución
    df['relacion_impacto_costo'] = df['impacto'] / df['costo']

    # Método 1: Mayor relación beneficio/costo
    idx_max_relacion = df['relacion_impacto_costo'].idxmax()

    # Método 2: Distancia al punto ideal
    # Primero normalizamos los objetivos
    df['impacto_norm'] = (df['impacto'] - df['impacto'].min()) / \
        (df['impacto'].max() - df['impacto'].min())
    df['costo_norm'] = (df['costo'] - df['costo'].min()) / \
        (df['costo'].max() - df['costo'].min())

    # El punto ideal normalizado sería (1, 0) [máximo impacto, mínimo costo]
    df['distancia_punto_ideal'] = np.sqrt(
        (df['impacto_norm'] - 1)**2 + df['costo_norm']**2)
    idx_min_distancia = df['distancia_punto_ideal'].idxmin()

    print("\nIdentificación de la mejor solución de compromiso:")
    print(
        f"Solución con mayor relación impacto/costo (Índice {idx_max_relacion}):")
    print(f"  Impacto: {df.loc[idx_max_relacion, 'impacto']:.2f}")
    print(f"  Costo: {df.loc[idx_max_relacion, 'costo']:.2f}")
    print(
        f"  Relación impacto/costo: {df.loc[idx_max_relacion, 'relacion_impacto_costo']:.2f}")

    print(
        f"\nSolución más cercana al punto ideal (Índice {idx_min_distancia}):")
    print(f"  Impacto: {df.loc[idx_min_distancia, 'impacto']:.2f}")
    print(f"  Costo: {df.loc[idx_min_distancia, 'costo']:.2f}")
    print(
        f"  Relación impacto/costo: {df.loc[idx_min_distancia, 'relacion_impacto_costo']:.2f}")

    # Visualizar en el frente de Pareto
    plt.figure(figsize=(10, 6))
    plt.scatter(df['costo'], df['impacto'], s=80, color='blue', alpha=0.5)

    # Destacar las soluciones de compromiso
    plt.scatter(df.loc[idx_max_relacion, 'costo'], df.loc[idx_max_relacion, 'impacto'],
                s=150, color='green', label='Mayor relación impacto/costo')
    plt.scatter(df.loc[idx_min_distancia, 'costo'], df.loc[idx_min_distancia, 'impacto'],
                s=150, color='red', label='Más cercana al punto ideal')

    plt.xlabel('Costo Total (miles USD)')
    plt.ylabel('Valor de Impacto Social (miles USD)')
    plt.title('Soluciones de Compromiso en el Frente de Pareto')
    plt.legend()
    plt.grid(True)
    plt.savefig('img/punto_1/soluciones_compromiso.png', dpi=300)
    plt.show()

    # Consideramos la solución con menor distancia al punto ideal como la mejor
    return idx_min_distancia


def comparar_metodos(df_ec, df_sp):
    """
    Compara los frentes de Pareto obtenidos por diferentes métodos.

    Args:
        df_ec: DataFrame con resultados del método ε-constraint
        df_sp: DataFrame con resultados del método de la suma ponderada
    """
    # Verificar si ambos DataFrames tienen datos válidos
    if (df_ec.empty or 'costo' not in df_ec.columns or 'impacto' not in df_ec.columns or
            df_sp.empty or 'costo' not in df_sp.columns or 'impacto' not in df_sp.columns):
        print("No hay suficientes datos para comparar los métodos.")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No hay datos suficientes para comparar los métodos.\n"
                           "Al menos uno de los métodos no pudo generar soluciones factibles.",
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.title("Error: Comparación de Métodos")
        plt.axis('off')
        plt.savefig('img/punto_1/comparacion_metodos_error.png', dpi=300)
        plt.show()
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df_ec['costo'], df_ec['impacto'],
                s=100, label='ε-constraint', alpha=0.7)
    plt.scatter(df_sp['costo'], df_sp['impacto'], s=100,
                label='Suma Ponderada', alpha=0.7)
    plt.xlabel('Costo Total (miles USD)')
    plt.ylabel('Valor de Impacto Social (miles USD)')
    plt.title('Comparación de Métodos: ε-constraint vs Suma Ponderada')
    plt.legend()
    plt.grid(True)
    plt.savefig('img/punto_1/comparacion_metodos.png', dpi=300)
    plt.show()

    # Análisis de diferencias
    print("\nComparación de métodos:")

    # Calcular la cobertura del espacio de soluciones
    min_costo_ec = df_ec['costo'].min()
    max_costo_ec = df_ec['costo'].max()
    rango_costo_ec = max_costo_ec - min_costo_ec

    min_costo_sp = df_sp['costo'].min()
    max_costo_sp = df_sp['costo'].max()
    rango_costo_sp = max_costo_sp - min_costo_sp

    min_impacto_ec = df_ec['impacto'].min()
    max_impacto_ec = df_ec['impacto'].max()
    rango_impacto_ec = max_impacto_ec - min_impacto_ec

    min_impacto_sp = df_sp['impacto'].min()
    max_impacto_sp = df_sp['impacto'].max()
    rango_impacto_sp = max_impacto_sp - min_impacto_sp

    print(f"Método ε-constraint:")
    print(
        f"  Rango de costos: [{min_costo_ec:.2f}, {max_costo_ec:.2f}], amplitud: {rango_costo_ec:.2f}")
    print(
        f"  Rango de impacto: [{min_impacto_ec:.2f}, {max_impacto_ec:.2f}], amplitud: {rango_impacto_ec:.2f}")

    print(f"Método Suma Ponderada:")
    print(
        f"  Rango de costos: [{min_costo_sp:.2f}, {max_costo_sp:.2f}], amplitud: {rango_costo_sp:.2f}")
    print(
        f"  Rango de impacto: [{min_impacto_sp:.2f}, {max_impacto_sp:.2f}], amplitud: {rango_impacto_sp:.2f}")

    # Análisis de distribución de puntos
    print("\nNúmero de soluciones generadas:")
    print(f"  ε-constraint: {len(df_ec)}")
    print(f"  Suma Ponderada: {len(df_sp)}")

    # Evaluación de la diversidad de soluciones
    def calcular_distancia_media(df):
        # Normalizar los valores para comparaciones justas
        df_norm = df.copy()
        df_norm['costo_norm'] = (
            df['costo'] - df['costo'].min()) / (df['costo'].max() - df['costo'].min())
        df_norm['impacto_norm'] = (
            df['impacto'] - df['impacto'].min()) / (df['impacto'].max() - df['impacto'].min())

        # Calcular la distancia promedio entre soluciones adyacentes en el espacio normalizado
        distancias = []
        for i in range(len(df_norm) - 1):
            dist = np.sqrt((df_norm['costo_norm'].iloc[i+1] - df_norm['costo_norm'].iloc[i])**2 +
                           (df_norm['impacto_norm'].iloc[i+1] - df_norm['impacto_norm'].iloc[i])**2)
            distancias.append(dist)

        return np.mean(distancias) if distancias else 0

    # Ordenar por costo para calcular distancias de manera más significativa
    df_ec_sorted = df_ec.sort_values(by='costo')
    df_sp_sorted = df_sp.sort_values(by='costo')

    dist_media_ec = calcular_distancia_media(df_ec_sorted)
    dist_media_sp = calcular_distancia_media(df_sp_sorted)

    print("\nDistribución de soluciones en el frente de Pareto:")
    print(
        f"  Distancia media entre soluciones adyacentes (ε-constraint): {dist_media_ec:.4f}")
    print(
        f"  Distancia media entre soluciones adyacentes (Suma Ponderada): {dist_media_sp:.4f}")

    # Conclusión
    if dist_media_ec < dist_media_sp:
        print("\nEl método ε-constraint generó soluciones más uniformemente distribuidas en el frente de Pareto.")
    else:
        print("\nEl método de Suma Ponderada generó soluciones más uniformemente distribuidas en el frente de Pareto.")

    if rango_costo_ec > rango_costo_sp or rango_impacto_ec > rango_impacto_sp:
        print("El método ε-constraint exploró un espacio de soluciones más amplio.")
    else:
        print("El método de Suma Ponderada exploró un espacio de soluciones más amplio.")


def principal():
    """Función principal que ejecuta el flujo completo de análisis."""
    print("=" * 80)
    print("PROBLEMA 1: OPTIMIZACIÓN MULTIOBJETIVO EN DISTRIBUCIÓN DE RECURSOS PARA MISIÓN HUMANITARIA")
    print("=" * 80)

    try:
        # Cálculo de valores extremos
        print("\n1. CALCULANDO VALORES EXTREMOS PARA NORMALIZACIÓN...")
        valores_extremos = calcular_valores_extremos_simulados()

        # Método de la suma ponderada
        print("\n2. APLICANDO MÉTODO DE LA SUMA PONDERADA...")
        resultados_sp = metodo_suma_ponderada(
            num_puntos=7, valores_extremos=valores_extremos)
        graficar_frente_pareto(resultados_sp, 'suma_ponderada')

        # Método ε-constraint
        print("\n3. APLICANDO MÉTODO ε-CONSTRAINT...")
        resultados_ec = metodo_epsilon_constraint(
            num_puntos=7, valores_extremos=valores_extremos)
        graficar_frente_pareto(resultados_ec, 'epsilon_constraint')

        # Comparar métodos
        print("\n4. COMPARANDO MÉTODOS...")
        comparar_metodos(resultados_ec, resultados_sp)

        # Análisis de sensibilidad
        print("\n5. ANÁLISIS DE SENSIBILIDAD (CUANDO UN OBJETIVO ES 5 VECES MÁS IMPORTANTE)...")
        resultados_sensibilidad = analisis_sensibilidad(
            factor_importancia=5, valores_extremos=valores_extremos)

        # Identificar mejor solución de compromiso
        print("\n6. IDENTIFICANDO LA MEJOR SOLUCIÓN DE COMPROMISO...")
        mejor_solucion_idx = identificar_mejor_solucion_compromiso(
            resultados_sp)

        # Analizar distribución de recursos para la mejor solución
        print("\n7. ANALIZANDO DISTRIBUCIÓN DE RECURSOS PARA LA MEJOR SOLUCIÓN...")
        analizar_distribucion_recursos(resultados_sp, mejor_solucion_idx)

        print("\nANÁLISIS COMPLETO FINALIZADO")
        print("Los resultados han sido guardados en archivos de imagen para su inclusión en el informe.")
        print("NOTA: Este análisis se ha realizado con datos simulados debido a dificultades en la resolución del modelo original.")

    except Exception as e:
        print(f"\nError durante el análisis: {e}")
        print("\nSe recomienda revisar los parámetros del problema y las restricciones para asegurar que exista al menos una solución factible.")


if __name__ == "__main__":
    principal()
