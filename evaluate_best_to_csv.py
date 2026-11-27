
"""
Evalúa modelos de 2048 y guarda CSV con las métricas.

Puede usarse de dos formas:

1) Como módulo (lo que hace main.py):
       import evaluate_best_to_csv as ev
       filas = ev.evaluar_variante("ppo_raw_s0", "models/ppo_raw_s0", 200)
       resumen = ev.resumir(filas)

2) Como script independiente:
       python evaluate_best_to_csv.py --episodes 200 \
           --variants "ppo_raw_s0=models/ppo_raw_s0,a2c_raw_s0=models/a2c_raw_s0"

En ambos casos, todos los CSVs van a la carpeta results/.
"""

import argparse
import csv
import os
import statistics
import sys
from typing import List, Tuple

import numpy as np

from enviroment import Game2048Env


CARPETA_RESULTS = "results"


# ============================================================
# CARGA DE MODELOS SEGÚN PREFIJO
# ============================================================

def cargar_modelo(nombre: str, ruta: str):
    """
    Carga el modelo adecuado según el prefijo del nombre de la variante.
    Devuelve (modelo, es_maskable).
    """
    prefijo = nombre.split("_")[0].lower()

    if prefijo == "ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(ruta, device="auto"), True

    if prefijo == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(ruta, device="auto"), False

    if prefijo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(ruta, device="auto"), False

    if prefijo == "qrdqn":
        from sb3_contrib import QRDQN
        return QRDQN.load(ruta, device="auto"), False

    raise ValueError(f"Algoritmo desconocido en variante '{nombre}'")


# ============================================================
# EVALUACIÓN DE UNA VARIANTE
# ============================================================

def evaluar_variante(
    nombre: str,
    ruta: str,
    episodes: int,
    seed: int = 1234,
) -> List[dict]:
    """
    Evalúa un modelo durante `episodes` episodios con acción determinista.
    Devuelve una lista de diccionarios, una fila por episodio.

    Todos evaluamos con target_tile=2048 y reward_type="raw", para que
    la comparación sea justa entre variantes con distinta recompensa de
    entrenamiento.
    """
    modelo, es_maskable = cargar_modelo(nombre, ruta)

    env = Game2048Env(
        size=4,
        target_tile=2048,
        observation_type="log2",
        reward_type="raw",
        invalid_penalty=1.0,
        max_episode_steps=2000,
    )

    filas = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = trunc = False
        total_reward = 0.0
        steps = 0

        while not (done or trunc):
            if es_maskable:
                am = env.action_masks()
                action, _ = modelo.predict(obs, deterministic=True, action_masks=am)
            else:
                action, _ = modelo.predict(obs, deterministic=True)

            obs, reward, done, trunc, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1

        filas.append({
            "variante": nombre,
            "episodio": ep,
            "recompensa": total_reward,
            "pasos": steps,
            "max_tile": int(info.get("max_tile", 0)),
            "gano": int(bool(info.get("won", False))),
            "puntuacion": int(info.get("score", 0)),
        })

    env.close()
    return filas


# ============================================================
# RESUMEN AGREGADO
# ============================================================

def resumir(filas: List[dict]) -> List[dict]:
    por_var: dict[str, list[dict]] = {}
    for f in filas:
        por_var.setdefault(f["variante"], []).append(f)

    resumen = []
    for nombre, grupo in por_var.items():
        recompensas = [g["recompensa"] for g in grupo]
        pasos = [g["pasos"] for g in grupo]
        max_tiles = [g["max_tile"] for g in grupo]
        gano = [g["gano"] for g in grupo]
        puntuacion = [g["puntuacion"] for g in grupo]

        resumen.append({
            "variante": nombre,
            "episodios": len(grupo),
            "recompensa_media": statistics.mean(recompensas),
            "recompensa_std": statistics.pstdev(recompensas)
                if len(recompensas) > 1 else 0.0,
            "pasos_media": statistics.mean(pasos),
            "max_tile_media": statistics.mean(max_tiles),
            "max_tile_max": max(max_tiles),
            "tasa_ganado": sum(gano) / len(gano),
            "puntuacion_media": statistics.mean(puntuacion),
        })
    return resumen


# ============================================================
# MODO SCRIPT (evaluación puntual)
# ============================================================

def parsear_variantes(cadena: str) -> List[Tuple[str, str]]:
    variantes = []
    for item in cadena.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Variante mal formada: {item}")
        nombre, ruta = item.split("=", 1)
        if ruta.endswith(".zip"):
            ruta = ruta[:-4]
        variantes.append((nombre.strip(), ruta.strip()))
    return variantes


def main():
    ap = argparse.ArgumentParser(
        description="Evalúa modelos de 2048 y guarda CSV en results/."
    )
    ap.add_argument("--episodes", type=int, default=200,
                    help="Episodios de evaluación por variante.")
    ap.add_argument("--variants", type=str, required=True,
                    help='Lista "nombre=ruta,nombre=ruta,..."')
    ap.add_argument("--out", type=str,
                    default=os.path.join(CARPETA_RESULTS, "eval_resultados.csv"),
                    help="CSV por episodio.")
    ap.add_argument("--out-summary", type=str,
                    default=os.path.join(CARPETA_RESULTS, "eval_resumen.csv"),
                    help="CSV resumen.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)

    pares = parsear_variantes(args.variants)
    print(f"Variantes a evaluar: {len(pares)}")
    for n, r in pares:
        print(f"  - {n}  ->  {r}")

    todas: List[dict] = []
    for nombre, ruta in pares:
        if not os.path.exists(ruta + ".zip") and not os.path.exists(ruta):
            print(f"[AVISO] No existe '{ruta}'. Se omite.", file=sys.stderr)
            continue
        print(f"\nEvaluando {nombre} ...")
        filas = evaluar_variante(nombre, ruta, args.episodes, args.seed)
        todas.extend(filas)
        print(f"  episodios: {len(filas)}")

    # CSV por episodio
    campos = ["variante", "episodio", "recompensa", "pasos",
              "max_tile", "gano", "puntuacion"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for fila in todas:
            w.writerow(fila)
    print(f"\nCSV por episodio: {args.out}")

    # CSV resumen
    resumen = resumir(todas)
    campos_resumen = [
        "variante", "episodios",
        "recompensa_media", "recompensa_std",
        "pasos_media", "max_tile_media", "max_tile_max",
        "tasa_ganado", "puntuacion_media",
    ]
    with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos_resumen)
        w.writeheader()
        for fila in resumen:
            w.writerow(fila)
    print(f"CSV resumen:      {args.out_summary}")

    # Tabla consola
    print("\n" + "=" * 90)
    print(f"{'variante':<40} {'recomp.':>10} {'max_tile':>10} {'gano':>8}")
    print("=" * 90)
    for r in sorted(resumen, key=lambda x: x["recompensa_media"], reverse=True):
        print(f"{r['variante']:<40} "
              f"{r['recompensa_media']:>10.2f} "
              f"{r['max_tile_media']:>10.1f} "
              f"{r['tasa_ganado']*100:>7.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()