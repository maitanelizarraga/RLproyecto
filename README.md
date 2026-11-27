# 2048 — Entrenamiento con MaskablePPO y Curriculum Learning

Entorno de 2048 implementado como `gymnasium.Env` y entrenado con
`MaskablePPO` (sb3-contrib). Soporta entrenamiento **con** y **sin**
curriculum learning, y dos tipos de recompensa: `raw` y `combined`.

---

## 1. Requisitos

- Python 3.10+
- Dependencias:

```bash
pip install gymnasium stable-baselines3 sb3-contrib tensorboard numpy# 2048 RL — Entrenamiento con MaskablePPO

