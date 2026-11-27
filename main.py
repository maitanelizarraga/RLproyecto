"""
Lanza todos los experimentos del proyecto desde un solo sitio.

Cada ejecución entrena UN algoritmo con UN tipo de recompensa de principio
a fin ("raw" o "combined"), con o sin curriculum (target_tile 128→512→2048).
Matriz completa: algoritmos x recompensas x {sin curriculum, con curriculum} x semillas.

Colocación esperada:
    proyecto/
        main.py
        enviroment.py
        models/ A2C.py  DQN.py  QR-DQN.py  PPO.py

Ejemplos:
    python main.py --dry-run                         # solo muestra el plan
    python main.py --timesteps 50000                 # prueba corta, todo
    python main.py --algos ppo dqn --rewards raw     # un subconjunto
    python main.py --curriculum si --seeds 0 1 2     # solo con curriculum
    python main.py --timesteps 1000000 --seeds 0 1 2 # experimento completo

Si se interrumpe, al volver a lanzarlo se saltan las ejecuciones que ya
tienen su modelo final (models/<nombre>.zip). Usa --rehacer para repetirlas.
"""
import argparse
import importlib.util
import os
import time
import traceback

CARPETA_MODELOS = "models/"

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
    ruta = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        CARPETA_MODELOS,
        ALGORITMOS[algoritmo],
    )
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


def main():
    parser = argparse.ArgumentParser(
        description="Entrena todos los algoritmos con ambas recompensas, "
                    "con y sin curriculum."
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
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--invalid-penalty", type=float, default=1.0,
                        help="Penalización por movimiento inválido en los "
                             "algoritmos SIN máscara (A2C, DQN, QR-DQN).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra el plan, no entrena.")
    parser.add_argument("--rehacer", action="store_true",
                        help="Repite también las ejecuciones ya terminadas.")
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

    modulos = {}
    resultados = []   

    for i, (algoritmo, recompensa, curriculum, semilla) in enumerate(plan, 1):
        if algoritmo not in modulos:
            modulos[algoritmo] = cargar_modulo(algoritmo) if not args.dry_run else None
        modulo = modulos[algoritmo]

        if modulo is not None:
            nombre = modulo.nombre_ejecucion(recompensa, curriculum, semilla)
        else:
            modo = "curriculum" if curriculum else "sin_curriculum"
            nombre = f"{algoritmo}_{recompensa}_{modo}_s{semilla}"

        etiqueta = f"[{i}/{len(plan)}] {nombre}"

        if args.dry_run:
            print(f"  {etiqueta}")
            continue

        if (not args.rehacer
                and os.path.exists(os.path.join(CARPETA_MODELOS, nombre + ".zip"))):
            print(f"{etiqueta}: ya existe, se salta")
            resultados.append((nombre, "saltada", 0.0))
            continue

        print("\n" + "#" * 70)
        print(f"# {etiqueta}")
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
        except Exception:
            traceback.print_exc()
            resultados.append((nombre, "ERROR", time.time() - inicio))

    if args.dry_run:
        return

    print("\n" + "=" * 70)
    print(" RESUMEN")
    print("=" * 70)
    for nombre, estado, segundos in resultados:
        print(f"  {estado:<8} {nombre:<45} {segundos / 60:7.1f} min")
    errores = [r for r in resultados if r[1] == "ERROR"]
    if errores:
        print(f"\n{len(errores)} ejecución(es) con error: revisa el traceback de arriba.")


if __name__ == "__main__":
    main()