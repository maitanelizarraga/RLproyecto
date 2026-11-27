"""
Lanza todo el proyecto desde un solo sitio:

    1. ENTRENA  cada algoritmo con cada recompensa, con y sin curriculum.
    2. EVALÚA   los modelos entrenados y guarda los CSV en results/.
    3. DIBUJA   las gráficas comparativas en results/plots/.

Cada entrenamiento usa UN tipo de recompensa ("raw" o "combined") de principio
a fin, con o sin curriculum (target_tile 128 -> 512 -> 2048).

Estructura esperada:
    proyecto/
        main.py
        enviroment.py
        evaluate_best_to_csv.py
        plot_comparison.py
        models/                 <- código de los algoritmos
            A2C.py  DQN.py  QR-DQN.py  PPO.py
            results/            <- modelos entrenados (se crea sola)
        logs/                   <- TensorBoard (se crea sola)
        results/                <- CSV y gráficas (se crea sola)

Ejemplos:
    python main.py --timesteps 50000 --episodes 20     # prueba corta de todo
    python main.py --timesteps 1000000 --seeds 0 1 2   # experimento completo
    python main.py --algos ppo dqn --rewards raw       # un subconjunto
"""
import argparse
import importlib.util
import os
import sys
import time
import traceback

# Todas las rutas son relativas a la carpeta de main.py
RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(RAIZ)
sys.path.insert(0, RAIZ)      # para importar enviroment.py, evaluate_best_to_csv.py...

CARPETA_ALGORITMOS = "models"            # aquí está el código de los algoritmos
CARPETA_MODELOS = "models/results"       # aquí se guardan los modelos entrenados
CARPETA_RESULTS = "results"              # CSV y gráficas

# nombre corto -> archivo (QR-DQN.py lleva guion, así que se carga por ruta)
ALGORITMOS = {
    "a2c": "A2C.py",
    "dqn": "DQN.py",
    "qrdqn": "QR-DQN.py",
    "ppo": "PPO.py",      # MaskablePPO
}
RECOMPENSAS = ["raw", "combined"]


def cargar_modulo(algoritmo):
    """Importa models/<archivo>.py aunque el nombre no sea un identificador válido."""
    ruta = os.path.join(RAIZ, CARPETA_ALGORITMOS, ALGORITMOS[algoritmo])
    spec = importlib.util.spec_from_file_location(f"algo_{algoritmo}", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def construir_plan(algoritmos, recompensas, curriculums, semillas):
    """Semilla en el bucle exterior: si falta tiempo, queda completa la matriz
    de las primeras semillas (todos los algoritmos y configuraciones)."""
    plan = []
    for semilla in semillas:
        for curriculum in curriculums:
            for recompensa in recompensas:
                for algoritmo in algoritmos:
                    plan.append((algoritmo, recompensa, curriculum, semilla))
    return plan


# ============================================================
# 1) ENTRENAMIENTO
# ============================================================

def entrenar_todo(plan, args):
    """Entrena cada ejecución del plan. Devuelve la lista de nombres de las
    que han terminado bien."""
    modulos = {}
    resultados = []   # (nombre, estado, segundos)
    terminadas = []

    for i, (algoritmo, recompensa, curriculum, semilla) in enumerate(plan, 1):
        if algoritmo not in modulos:
            modulos[algoritmo] = cargar_modulo(algoritmo)
        modulo = modulos[algoritmo]

        nombre = modulo.nombre_ejecucion(recompensa, curriculum, semilla)

        print("\n" + "#" * 70)
        print(f"# [{i}/{len(plan)}] {nombre}")
        print("#" * 70)

        inicio = time.time()
        try:
            modulo.entrenar(
                timesteps=args.timesteps,
                eval_freq=args.eval_freq,
                seed=semilla,
                curriculum=curriculum,
                reward_type=recompensa,
                invalid_penalty=args.invalid_penalty,
            )
            resultados.append((nombre, "ok", time.time() - inicio))
            terminadas.append(nombre)
        except Exception:
            traceback.print_exc()
            resultados.append((nombre, "ERROR", time.time() - inicio))

    print("\n" + "=" * 70)
    print(" RESUMEN DEL ENTRENAMIENTO")
    print("=" * 70)
    for nombre, estado, segundos in resultados:
        print(f"  {estado:<8} {nombre:<45} {segundos / 60:7.1f} min")

    return terminadas


# ============================================================
# 2) EVALUACIÓN
# ============================================================

def evaluar_todo(nombres, episodios):
    """Evalúa los modelos entrenados y guarda los CSV en results/."""
    import evaluate_best_to_csv as ev

    print("\n" + "=" * 70)
    print(f" EVALUACIÓN: {len(nombres)} modelos x {episodios} episodios")
    print("=" * 70)

    filas = []
    for nombre in nombres:
        ruta = os.path.join(CARPETA_MODELOS, nombre)
        print(f"Evaluando {nombre} ...")
        filas.extend(ev.evaluar_variante(nombre, ruta, episodios))

    resumen = ev.resumir(filas)

    ruta_episodios = os.path.join(CARPETA_RESULTS, "eval_resultados.csv")
    ruta_resumen = os.path.join(CARPETA_RESULTS, "eval_resumen.csv")
    ev.guardar_csv(filas, resumen, ruta_episodios, ruta_resumen)
    print(f"\nCSV por episodio: {ruta_episodios}")
    print(f"CSV resumen:      {ruta_resumen}")

    print("\n" + "=" * 90)
    print(f"{'variante':<40} {'recomp.':>10} {'max_tile':>10} {'gano':>8}")
    print("=" * 90)
    for r in sorted(resumen, key=lambda x: x["recompensa_media"], reverse=True):
        print(f"{r['variante']:<40} "
              f"{r['recompensa_media']:>10.2f} "
              f"{r['max_tile_media']:>10.1f} "
              f"{r['tasa_ganado'] * 100:>7.1f}%")
    print("=" * 90)

    return ruta_resumen, ruta_episodios


# ============================================================
# 3) GRÁFICAS
# ============================================================

def dibujar_todo(ruta_resumen, ruta_episodios):
    import plot_comparison as pc

    print("\n" + "=" * 70)
    print(" GRÁFICAS")
    print("=" * 70)
    pc.generar_plots(ruta_resumen, ruta_episodios,
                     os.path.join(CARPETA_RESULTS, "plots"))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entrena, evalúa y dibuja: todos los algoritmos con ambas "
                    "recompensas, con y sin curriculum."
    )
    parser.add_argument("--algos", nargs="+", choices=list(ALGORITMOS),
                        default=list(ALGORITMOS))
    parser.add_argument("--rewards", nargs="+", choices=RECOMPENSAS,
                        default=RECOMPENSAS)
    parser.add_argument("--curriculum", choices=["si", "no", "ambos"],
                        default="ambos")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Timesteps TOTALES de cada ejecución.")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                        help="Frecuencia de evaluación durante el entrenamiento.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Episodios de la evaluación final de cada modelo.")
    parser.add_argument("--invalid-penalty", type=float, default=1.0,
                        help="Penalización por movimiento inválido en los "
                             "algoritmos SIN máscara (A2C, DQN, QR-DQN).")
    args = parser.parse_args()

    curriculums = {"si": [True], "no": [False], "ambos": [False, True]}[args.curriculum]
    plan = construir_plan(args.algos, args.rewards, curriculums, args.seeds)

    print("=" * 70)
    print(f" PLAN: {len(plan)} ejecuciones x {args.timesteps:,} timesteps "
          f"= {len(plan) * args.timesteps:,} timesteps en total")
    print(f"   algoritmos:  {args.algos}")
    print(f"   recompensas: {args.rewards}")
    print(f"   curriculum:  {args.curriculum}")
    print(f"   semillas:    {args.seeds}")
    print("=" * 70)

    terminadas = entrenar_todo(plan, args)

    if not terminadas:
        print("\nNingún entrenamiento terminó bien: no hay nada que evaluar.")
        return

    ruta_resumen, ruta_episodios = evaluar_todo(terminadas, args.episodes)
    dibujar_todo(ruta_resumen, ruta_episodios)


if __name__ == "__main__":
    main()