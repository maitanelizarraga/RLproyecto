# plot_comparison.py
"""
Genera gráficas comparativas a partir de los CSV de evaluación.

Lee por defecto:
    results/eval_resultados.csv   (una fila por episodio)
    results/eval_resumen.csv      (media/std por variante)

Produce (en results/plots/):
    1.  comparacion_recompensa.png
    2.  comparacion_max_tile.png
    3.  comparacion_tasa_ganado.png
    4.  comparacion_pasos.png
    5.  boxplot_max_tile.png
    6.  curva_aprendizaje.png
    7.  heatmap_recompensa.png
    8.  comparacion_por_algoritmo.png
    9.  comparacion_por_curriculum.png
   10.  heatmap_reward_curriculum.png

Los nombres de las variantes siguen el patrón:
    {algo}_{reward}_{sin_curriculum|curriculum}_s{seed}
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CARPETA_RESULTS = "results"


# ============================================================
# CARGA DE CSVs
# ============================================================

def cargar_resumen(ruta: str) -> List[dict]:
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes la evaluación."
        )
    with open(ruta, newline="", encoding="utf-8") as f:
        return [
            {k: (float(v) if _es_numero(v) else v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def cargar_episodios(ruta: str) -> List[dict]:
    if not os.path.exists(ruta):
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = []
        for row in csv.DictReader(f):
            filas.append({
                "variante": row["variante"],
                "episodio": int(row["episodio"]),
                "recompensa": float(row["recompensa"]),
                "pasos": int(row["pasos"]),
                "max_tile": int(row["max_tile"]),
                "gano": int(row["gano"]),
                "puntuacion": int(row["puntuacion"]),
            })
        return filas


def _es_numero(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


# ============================================================
# PARSEO DEL NOMBRE DE VARIANTE
# ============================================================

_RE = re.compile(
    r"^(?P<algo>[a-z0-9]+)"
    r"_(?P<reward>raw|combined)"
    r"_(?P<modo>sin_curriculum|curriculum)"
    r"_s(?P<seed>\d+)$"
)


def parsear_nombre(nombre: str) -> dict:
    m = _RE.match(nombre)
    if not m:
        return {"algo": "?", "reward": "?", "modo": "?", "seed": 0}
    return {
        "algo": m.group("algo"),
        "reward": m.group("reward"),
        "modo": m.group("modo"),
        "seed": int(m.group("seed")),
    }


def enriquecer(resumen: List[dict]) -> List[dict]:
    for r in resumen:
        r.update(parsear_nombre(r["variante"]))
    return resumen


# ============================================================
# UTILIDADES DE PLOT
# ============================================================

def _colores(n: int) -> List:
    base = plt.get_cmap("tab10").colors
    return [base[i % len(base)] for i in range(n)]


def _media_std(resumen, algo, modo, reward):
    """Media de la recompensa de las filas (semillas) de una combinación.
    Barra de error: desviación entre semillas si hay varias; si solo hay
    una, la desviación entre episodios de esa variante."""
    filas = [r for r in resumen
             if r["algo"] == algo and r["modo"] == modo and r["reward"] == reward]
    if not filas:
        return 0.0, 0.0
    medias = [r["recompensa_media"] for r in filas]
    if len(filas) == 1:
        return medias[0], filas[0]["recompensa_std"]
    return float(np.mean(medias)), float(np.std(medias))


def _guardar(fig, carpeta: str, nombre: str):
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    print(f"  guardado: {ruta}")


# ============================================================
# PLOTS
# ============================================================

def plot_barras_metrica(resumen, metrica, ylabel, titulo, carpeta, fichero,
                        fmt=".2f", error_col=None):
    datos = sorted(resumen, key=lambda r: r[metrica], reverse=True)
    nombres = [r["variante"] for r in datos]
    valores = [r[metrica] for r in datos]
    errores = [r[error_col] for r in datos] if error_col else None

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(nombres) + 4), 5))
    colores = _colores(len(nombres))
    bars = ax.bar(range(len(nombres)), valores, color=colores,
                  yerr=errores, capsize=4 if errores else 0)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, valores):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=7)
    _guardar(fig, carpeta, fichero)


def plot_boxplot_max_tile(episodios, carpeta):
    if not episodios:
        print("  [skip] sin CSV por episodio para boxplot")
        return
    por_var: Dict[str, List[int]] = defaultdict(list)
    for e in episodios:
        por_var[e["variante"]].append(e["max_tile"])
    nombres = sorted(por_var.keys())
    datos = [por_var[n] for n in nombres]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(nombres) + 4), 5))
    ax.boxplot(datos, showmeans=True)
    ax.set_xticks(range(1, len(nombres) + 1))
    ax.set_xticklabels(nombres, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Máxima ficha alcanzada")
    ax.set_title("Distribución de la ficha máxima por variante")
    ax.grid(axis="y", alpha=0.3)
    _guardar(fig, carpeta, "boxplot_max_tile.png")


def plot_curva_aprendizaje(episodios, carpeta):
    if not episodios:
        print("  [skip] sin CSV por episodio para curva")
        return
    por_var: Dict[str, List[float]] = defaultdict(list)
    for e in episodios:
        por_var[e["variante"]].append(e["recompensa"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for nombre, vals in sorted(por_var.items()):
        acum = np.cumsum(vals) / (np.arange(len(vals)) + 1)
        ax.plot(acum, label=nombre, linewidth=1.2)
    ax.set_xlabel("Episodio de evaluación")
    ax.set_ylabel("Recompensa media acumulada")
    ax.set_title("Evolución de la recompensa media durante la evaluación")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=3, loc="best")
    _guardar(fig, carpeta, "curva_aprendizaje.png")


def plot_heatmap_recompensa(resumen, carpeta):
    """Heatmap algoritmo x (reward+modo)."""
    algoritmos = sorted({r["algo"] for r in resumen})
    columnas = sorted({f"{r['modo']}|{r['reward']}" for r in resumen})

    tabla = np.full((len(algoritmos), len(columnas)), np.nan)
    for i, alg in enumerate(algoritmos):
        for j, col in enumerate(columnas):
            modo, reward = col.split("|")
            if any(r["algo"] == alg and r["modo"] == modo and r["reward"] == reward
                   for r in resumen):
                tabla[i, j] = _media_std(resumen, alg, modo, reward)[0]

    fig, ax = plt.subplots(
        figsize=(1.4 * len(columnas) + 4, 0.9 * len(algoritmos) + 3)
    )
    im = ax.imshow(tabla, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(columnas)))
    ax.set_xticklabels(columnas, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(algoritmos)))
    ax.set_yticklabels(algoritmos)
    ax.set_title("Recompensa media: algoritmo x (curriculum | reward)")
    for i in range(len(algoritmos)):
        for j in range(len(columnas)):
            v = tabla[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Recompensa media")
    _guardar(fig, carpeta, "heatmap_recompensa.png")


def plot_comparacion_por_algoritmo(resumen, carpeta):
    """
    Barplot agrupado: para cada algoritmo, una barra por cada combinación
    (modo, reward). Permite ver de un vistazo las 4 variantes de cada algo.
    """
    algoritmos = sorted({r["algo"] for r in resumen})
    combos = sorted({f"{r['modo']}|{r['reward']}" for r in resumen})

    x = np.arange(len(algoritmos))
    ancho = 0.8 / max(len(combos), 1)
    colores = _colores(len(combos))

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(algoritmos) + 4), 5))
    for k, combo in enumerate(combos):
        modo, reward = combo.split("|")
        valores, errores = [], []
        for alg in algoritmos:
            media, error = _media_std(resumen, alg, modo, reward)
            valores.append(media)
            errores.append(error)
        ax.bar(x + k * ancho - 0.4 + ancho / 2, valores, ancho,
               yerr=errores, capsize=3,
               label=f"{modo} / {reward}", color=colores[k])

    ax.set_xticks(x)
    ax.set_xticklabels(algoritmos)
    ax.set_ylabel("Recompensa media")
    ax.set_title("Comparación por algoritmo y combinación (modo, reward)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="modo / reward", fontsize=8)
    _guardar(fig, carpeta, "comparacion_por_algoritmo.png")


def plot_comparacion_por_curriculum(resumen, carpeta):
    """
    Para cada situación (sin_curriculum vs curriculum), compara
    raw vs combined por algoritmo. Dos subplots, uno por situación.
    """
    algoritmos = sorted({r["algo"] for r in resumen})
    rewards = sorted({r["reward"] for r in resumen})
    modos = ["sin_curriculum", "curriculum"]

    if not algoritmos or not rewards:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colores = {"raw": "#1f77b4", "combined": "#ff7f0e"}

    for ax, modo in zip(axes, modos):
        x = np.arange(len(algoritmos))
        ancho = 0.35
        for k, reward in enumerate(rewards):
            valores, errores = [], []
            for alg in algoritmos:
                media, error = _media_std(resumen, alg, modo, reward)
                valores.append(media)
                errores.append(error)
            ax.bar(x + k * ancho - ancho / 2, valores, ancho,
                   yerr=errores, capsize=3,
                   label=reward, color=colores.get(reward, None))
        ax.set_xticks(x)
        ax.set_xticklabels(algoritmos)
        ax.set_title(f"Modo: {modo}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="reward")

    axes[0].set_ylabel("Recompensa media")
    fig.suptitle("Raw vs combined dentro de cada situación (con/sin curriculum)")
    _guardar(fig, carpeta, "comparacion_por_curriculum.png")


def plot_heatmap_reward_curriculum(resumen, carpeta):
    """
    Heatmap algoritmo x (modo, reward) = 4 columnas.
    La vista más compacta para comparar las 4 combinaciones.
    """
    algoritmos = sorted({r["algo"] for r in resumen})
    columnas = [
        ("sin_curriculum", "raw"),
        ("sin_curriculum", "combined"),
        ("curriculum", "raw"),
        ("curriculum", "combined"),
    ]

    tabla = np.full((len(algoritmos), len(columnas)), np.nan)
    for i, alg in enumerate(algoritmos):
        for j, (modo, reward) in enumerate(columnas):
            if any(r["algo"] == alg and r["modo"] == modo and r["reward"] == reward
                   for r in resumen):
                tabla[i, j] = _media_std(resumen, alg, modo, reward)[0]

    fig, ax = plt.subplots(figsize=(1.6 * len(columnas) + 4,
                                    0.9 * len(algoritmos) + 3))
    im = ax.imshow(tabla, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(columnas)))
    ax.set_xticklabels([f"{m}\n{r}" for m, r in columnas], fontsize=8)
    ax.set_yticks(range(len(algoritmos)))
    ax.set_yticklabels(algoritmos)
    ax.set_title("Recompensa media: algoritmo x (modo, reward)")
    for i in range(len(algoritmos)):
        for j in range(len(columnas)):
            v = tabla[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Recompensa media")
    _guardar(fig, carpeta, "heatmap_reward_curriculum.png")


# ============================================================
# MAIN (modo script)
# ============================================================

def generar_plots(ruta_resumen: str, ruta_episodios: str, outdir: str):
    """Carga los CSV y genera todas las gráficas en `outdir`."""
    print("Cargando CSVs...")
    resumen = enriquecer(cargar_resumen(ruta_resumen))
    episodios = cargar_episodios(ruta_episodios)
    print(f"  variantes en resumen: {len(resumen)}")
    print(f"  filas por episodio:   {len(episodios)}")

    os.makedirs(outdir, exist_ok=True)

    print("\nGenerando plots...")
    plot_barras_metrica(resumen, "recompensa_media", "Recompensa media",
                        "Recompensa media por variante (mayor = mejor)",
                        outdir, "comparacion_recompensa.png",
                        fmt=".2f", error_col="recompensa_std")
    plot_barras_metrica(resumen, "max_tile_media", "Ficha máxima media",
                        "Ficha máxima media por variante (mayor = mejor)",
                        outdir, "comparacion_max_tile.png", fmt=".1f")
    plot_barras_metrica(resumen, "tasa_ganado", "Tasa de victoria",
                        "Porcentaje de partidas ganadas (llegar a 2048)",
                        outdir, "comparacion_tasa_ganado.png", fmt=".2%")
    plot_barras_metrica(resumen, "pasos_media", "Pasos medios",
                        "Duración media de las partidas",
                        outdir, "comparacion_pasos.png", fmt=".1f")
    plot_boxplot_max_tile(episodios, outdir)
    plot_curva_aprendizaje(episodios, outdir)
    plot_heatmap_recompensa(resumen, outdir)
    plot_comparacion_por_algoritmo(resumen, outdir)
    plot_comparacion_por_curriculum(resumen, outdir)
    plot_heatmap_reward_curriculum(resumen, outdir)

    print(f"\nListo. Figuras en {outdir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Genera plots comparativos a partir de los CSV de evaluación."
    )
    ap.add_argument("--results", type=str,
                    default=os.path.join(CARPETA_RESULTS, "eval_resultados.csv"))
    ap.add_argument("--summary", type=str,
                    default=os.path.join(CARPETA_RESULTS, "eval_resumen.csv"))
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(CARPETA_RESULTS, "plots"))
    args = ap.parse_args()

    generar_plots(args.summary, args.results, args.outdir)

if __name__ == "__main__":
    main()