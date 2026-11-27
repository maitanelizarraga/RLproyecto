import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Game2048Env(gym.Env):
    """
    Observation types:
        - raw: board values
        - log2: log2(board + 1)
        - one_hot: one channel per tile exponent

    Reward types (solo dos disponibles):
        - raw: score obtained from merges
        - combined: merge reward + bonus por casillas vacías
                    + bonus por mantener la ficha mayor en una esquina

    Actions:
        0 = left
        1 = right
        2 = up
        3 = down
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        size=4,
        target_tile=2048,

        observation_type="log2",
        reward_type="raw",          # solo "raw" o "combined"

        invalid_penalty=0.0,

        empty_bonus=0.1,
        corner_bonus=0.1,

        max_episode_steps=2000,

        render_mode=None
    ):
        super().__init__()

        assert size >= 2
        assert target_tile >= 2
        assert observation_type in ["raw", "log2", "one_hot"]

        # Solo dos tipos de reward permitidos
        assert reward_type in ["raw", "combined"], (
            f"reward_type debe ser 'raw' o 'combined', recibido: {reward_type}"
        )

        self.size = size
        self.target_tile = target_tile
        self.observation_type = observation_type
        self.reward_type = reward_type

        self.invalid_penalty = invalid_penalty
        self.empty_bonus = empty_bonus
        self.corner_bonus = corner_bonus

        self.max_episode_steps = max_episode_steps

        self.render_mode = render_mode

        self.board = None
        self.score = 0
        self.steps = 0
        self.total_steps = 0

        self.action_space = spaces.Discrete(4)

        if observation_type == "raw":
            self.observation_space = spaces.Box(
                low=0,
                high=np.iinfo(np.int32).max,
                shape=(size, size),
                dtype=np.int32
            )
        elif observation_type == "log2":
            self.observation_space = spaces.Box(
                low=0.0,
                high=31.0,
                shape=(size, size),
                dtype=np.float32
            )
        elif observation_type == "one_hot":
            self.num_channels = 12
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_channels, size, size),
                dtype=np.float32
            )

    # --------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------

    def _get_obs(self):
        if self.observation_type == "raw":
            return self.board.copy()

        if self.observation_type == "log2":
            return np.log2(
                self.board.astype(np.float32) + 1.0
            ).astype(np.float32)

        if self.observation_type == "one_hot":
            obs = np.zeros(
                (self.num_channels, self.size, self.size),
                dtype=np.float32
            )
            exponents = np.zeros_like(self.board, dtype=np.int32)
            nonzero = self.board > 0
            exponents[nonzero] = np.rint(
                np.log2(self.board[nonzero])
            ).astype(np.int32)
            exponents = np.clip(exponents, 0, self.num_channels - 1)
            rows, cols = np.indices((self.size, self.size))
            obs[exponents, rows, cols] = 1.0
            return obs

    # --------------------------------------------------
    # GAME LOGIC
    # --------------------------------------------------

    def _add_new_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            return
        idx = self.np_random.integers(len(empty_cells))
        row, col = empty_cells[idx]
        value = 2 if self.np_random.random() < 0.9 else 4
        self.board[row, col] = value

    def _merge(self, line):
        nonzero = line[line != 0]
        merged = []
        score = 0
        i = 0
        while i < len(nonzero):
            if (i + 1 < len(nonzero)
                    and nonzero[i] == nonzero[i + 1]):
                new_value = nonzero[i] * 2
                merged.append(new_value)
                score += new_value
                i += 2
            else:
                merged.append(nonzero[i])
                i += 1
        merged.extend([0] * (self.size - len(merged)))
        return np.array(merged, dtype=np.int32), score

    def _move_board(self, board, action):
        board = board.copy()

        if action == 0:
            pass
        elif action == 1:
            board = np.fliplr(board)
        elif action == 2:
            board = board.T
        elif action == 3:
            board = np.fliplr(board.T)

        total_score = 0
        for i in range(self.size):
            board[i], move_score = self._merge(board[i])
            total_score += move_score

        if action == 0:
            pass
        elif action == 1:
            board = np.fliplr(board)
        elif action == 2:
            board = board.T
        elif action == 3:
            board = np.fliplr(board).T

        return board.copy(), total_score

    def _move(self, action):
        new_board, score = self._move_board(self.board, action)
        changed = not np.array_equal(self.board, new_board)

        if changed:
            self.board = new_board
            self._add_new_tile()
            self.score += score
            self.steps += 1

        return score, changed

    # --------------------------------------------------
    # VALID ACTIONS / MASKING
    # --------------------------------------------------

    def action_masks(self):
        masks = np.zeros(4, dtype=bool)
        if self.board is None:
            return masks
        for action in range(4):
            new_board, _ = self._move_board(self.board, action)
            masks[action] = not np.array_equal(self.board, new_board)
        return masks

    def _is_game_over(self):
        return not np.any(self.action_masks())

    # --------------------------------------------------
    # REWARD SHAPING (solo raw y combined)
    # --------------------------------------------------

    def _get_reward(self, merge_score, valid_move):
        if not valid_move:
            return float(-self.invalid_penalty)

        if self.reward_type == "raw":
            return float(merge_score)

        # reward_type == "combined"
        empty = np.count_nonzero(self.board == 0)
        largest = np.max(self.board)
        corners = [
            self.board[0, 0],
            self.board[0, -1],
            self.board[-1, 0],
            self.board[-1, -1]
        ]
        reward = float(merge_score) + self.empty_bonus * empty
        if largest in corners:
            reward += self.corner_bonus
        return float(reward)

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros(
            (self.size, self.size),
            dtype=np.int32
        )
        self.score = 0
        self.steps = 0
        self.total_steps = 0

        self._add_new_tile()
        self._add_new_tile()

        info = {
            "score": self.score,
            "steps": self.steps,
            "max_tile": int(np.max(self.board))
        }
        return self._get_obs(), info

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        self.total_steps += 1
        merge_score, valid_move = self._move(action)
        reward = self._get_reward(merge_score, valid_move)

        terminated = (
            self._is_game_over()
            or np.max(self.board) >= self.target_tile
        )

        truncated = bool(
            self.total_steps >= self.max_episode_steps
            and not terminated
        )

        info = {
            "score": self.score,
            "steps": self.steps,
            "max_tile": int(np.max(self.board)),
            "empty_cells": int(np.count_nonzero(self.board == 0)),
            "valid_move": valid_move
        }

        return self._get_obs(), reward, terminated, truncated, info

    # --------------------------------------------------
    # RENDER
    # --------------------------------------------------

    def render(self):
        print("\n" + "=" * 30)
        print(self.board)
        print("=" * 30)
        print("Score:", self.score)
        print("Max tile:", np.max(self.board))
        print("Steps:", self.steps)

    def close(self):
        pass


# ======================================================
# COMPARACIÓN DE REWARDS: raw vs combined
# ======================================================

def evaluate_reward(
    reward_type,
    episodes=20,
    policy=None,
    seed=42,
    env_kwargs=None,
    verbose=False
):
    """
    Ejecuta `episodes` episodios con un reward concreto y devuelve métricas.

    Parámetros
    ----------
    reward_type : "raw" | "combined"
    episodes    : número de episodios a jugar
    policy      : callable(obs, action_masks) -> action
                  Si es None, se usa una política aleatoria válida.
    seed        : semilla base (se reparte entre episodios)
    env_kwargs  : kwargs extra para Game2048Env
    verbose     : imprime progreso por episodio
    """
    if env_kwargs is None:
        env_kwargs = {}

    env = Game2048Env(reward_type=reward_type, **env_kwargs)

    scores = []
    steps = []
    max_tiles = []
    total_rewards = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            masks = env.action_masks()

            if policy is None:
                # política aleatoria válida
                valid_actions = np.flatnonzero(masks)
                if len(valid_actions) == 0:
                    break
                action = int(np.random.choice(valid_actions))
            else:
                action = policy(obs, masks)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        scores.append(info["score"])
        steps.append(info["steps"])
        max_tiles.append(info["max_tile"])
        total_rewards.append(ep_reward)

        if verbose:
            print(f"  [{reward_type}] ep {ep+1:02d} | "
                  f"score={info['score']:5d} | "
                  f"max_tile={info['max_tile']:4d} | "
                  f"steps={info['steps']:4d} | "
                  f"R={ep_reward:8.2f}")

    env.close()

    return {
        "reward_type": reward_type,
        "episodes": episodes,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "max_tile_mean": float(np.mean(max_tiles)),
        "max_tile_max": int(np.max(max_tiles)),
        "steps_mean": float(np.mean(steps)),
        "reward_mean": float(np.mean(total_rewards)),
    }


def run_comparison(
    episodes=20,
    seed=42,
    policy=None,
    env_kwargs=None,
    verbose=True
):
    """
    Ejecuta `evaluate_reward` primero con 'raw' y luego con 'combined'
    usando la misma semilla y la misma política, y devuelve un dict
    con los resultados listos para comparar.
    """
    results = {}

    print("\n" + "=" * 55)
    print(" EJECUTANDO CON reward_type = 'raw' ")
    print("=" * 55)
    results["raw"] = evaluate_reward(
        reward_type="raw",
        episodes=episodes,
        policy=policy,
        seed=seed,
        env_kwargs=env_kwargs,
        verbose=verbose
    )

    print("\n" + "=" * 55)
    print(" EJECUTANDO CON reward_type = 'combined' ")
    print("=" * 55)
    results["combined"] = evaluate_reward(
        reward_type="combined",
        episodes=episodes,
        policy=policy,
        seed=seed,
        env_kwargs=env_kwargs,
        verbose=verbose
    )

    # Resumen comparativo
    print("\n" + "=" * 55)
    print(" RESUMEN COMPARATIVO ")
    print("=" * 55)
    header = f"{'Métrica':<20}{'raw':>15}{'combined':>15}"
    print(header)
    print("-" * len(header))
    for key in ["score_mean", "score_std", "max_tile_mean",
                "max_tile_max", "steps_mean", "reward_mean"]:
        print(f"{key:<20}"
              f"{results['raw'][key]:>15.2f}"
              f"{results['combined'][key]:>15.2f}")

    return results


if __name__ == "__main__":
    # Comparación con política aleatoria, 20 episodios, misma semilla
    run_comparison(
        episodes=20,
        seed=42,
        policy=None,
        env_kwargs=dict(
            size=4,
            target_tile=2048,
            observation_type="log2",
            max_episode_steps=2000,
        ),
        verbose=True
    )