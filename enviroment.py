import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Game2048Env(gym.Env):
    """
    Observation types:
        - raw: board values
        - log2: log2(board + 1)
        - one_hot: one channel per tile exponent

    Reward types:
        - raw: score obtained from merges
        - log: logarithmic merge reward
        - empty_cells: merge reward + bonus for empty cells
        - corner: merge reward + bonus for keeping the
                  largest tile in a corner
        - combined: combines empty cells and corner bonuses

    Actions:
        0 = left
        1 = right
        2 = up
        3 = down
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        size=4, #la tabla es 4x4 de origen
        target_tile=2048, #ficha objetivo a alcanzar y termina el juego

        observation_type="log2",
        reward_type="raw",

        #penalizacion por movimientos invalidos
        invalid_penalty=0.0,

        #bonus por dejar casillas vacias y que las ficha mayor esten en la esquina(mejor estrategia de juego)
        empty_bonus=0.1,
        corner_bonus=0.1,

        #limite de pasos por episodio (cuenta tambien los movimientos invalidos)
        max_episode_steps=2000,

        #visualización
        render_mode=None
    ):
        super().__init__()

        #compureba que los valores seleccionados para el nviroment sean validos
        assert size >= 2
        assert target_tile >= 2
        assert observation_type in ["raw", "log2", "one_hot"]
        assert reward_type in [
            "raw", "log", "empty_cells",
            "corner", "combined"
        ]

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
        self.total_steps = 0 #todos los pasos, validos e invalidos (para truncar el episodio)

        # Actions: left, right, up, down
        self.action_space = spaces.Discrete(4)

        # Observation spaces
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

            self.num_channels = 12 #una capa por posible valor(incluyendo vacio) siendo el max 2048

            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_channels, size, size),
                dtype=np.float32
            )

    
    # OBSERVATIONS
       
    #devuelve al agente la tabla según lo que se haya especificado
    def _get_obs(self):

        if self.observation_type == "raw":
            return self.board.copy()

        if self.observation_type == "log2":

            return np.log2(
                self.board.astype(np.float32) + 1.0 #suma para que el log no de error en valores cero
            ).astype(np.float32)

        if self.observation_type == "one_hot":

            obs = np.zeros(
                (self.num_channels, self.size, self.size),
                dtype=np.float32
            )

            exponents = np.zeros_like(
                self.board,
                dtype=np.int32
            )

            nonzero = self.board > 0

            exponents[nonzero] = np.rint(
                np.log2(self.board[nonzero])
            ).astype(np.int32)

            exponents = np.clip(
                exponents, 0, self.num_channels - 1
            )

            rows, cols = np.indices(
                (self.size, self.size)
            )

            obs[exponents, rows, cols] = 1.0

            return obs

    
    # GAME LOGIC
       
    #añade la nueva ficha en un espacio aleatorio de las celdas vacias (90 para valor 2 y 10 para el 4)
    def _add_new_tile(self):

        empty_cells = np.argwhere(self.board == 0)

        if len(empty_cells) == 0:
            return

        idx = self.np_random.integers(len(empty_cells))

        row, col = empty_cells[idx]

        value = 2 if self.np_random.random() < 0.9 else 4

        self.board[row, col] = value


    #funcion union d fichas
    def _merge(self, line):

        nonzero = line[line != 0] #detecta ceros en la linea

        merged = []
        score = 0 #puntos que va a generar la union en ese mov

        i = 0 #inidca que posicion estamos comprobando

        while i < len(nonzero):

            if (
                i + 1 < len(nonzero)
                and nonzero[i] == nonzero[i + 1]
            ):

                new_value = nonzero[i] * 2

                merged.append(new_value)
                score += new_value

                i += 2 #indica que si dos fichas se acaban de fusionar, estas no sue pueden usar pa volver a hacer una nueva fusion

            else:

                merged.append(nonzero[i])
                i += 1

        merged.extend(
            [0] * (self.size - len(merged))
        )

        return np.array(
            merged, dtype=np.int32
        ), score

    #para poder reutilizar el merge ya uqe esta solo definido hacia la izq
    def _move_board(self, board, action):

        board = board.copy()

        # Transform all movements into left movement
        if action == 0:
            pass

        elif action == 1:
            board = np.fliplr(board)

        elif action == 2:
            board = board.T

        elif action == 3:
            board = np.fliplr(board.T) #antes flipud(board.T), que mezclaba las columnas

        total_score = 0

        for i in range(self.size):

            board[i], move_score = self._merge(board[i])

            total_score += move_score

        # Reverse transformation
        if action == 0:
            pass

        elif action == 1:
            board = np.fliplr(board)

        elif action == 2:
            board = board.T

        elif action == 3:
            board = np.fliplr(board).T #inversa exacta de fliplr(board.T)

        return board.copy(), total_score


    #calcula como queda la tabla resultante y si es igual que al principio el mov pasa a no valido
    def _move(self, action):

        new_board, score = self._move_board(
            self.board, action
        )

        changed = not np.array_equal(
            self.board, new_board
        )

        if changed:

            self.board = new_board

            self._add_new_tile()

            self.score += score

            self.steps += 1

        return score, changed

    
    # VALID ACTIONS / MASKING
       
    def action_masks(self):

        masks = np.zeros(4, dtype=bool)

        if self.board is None:
            return masks

        for action in range(4):

            new_board, _ = self._move_board(
                self.board, action
            )

            masks[action] = not np.array_equal(
                self.board, new_board
            )

        return masks

    def _is_game_over(self):

        return not np.any(self.action_masks())

    
    # REWARD SHAPING
       
    def _get_reward(self, merge_score, valid_move):

        if not valid_move:
            return float(-self.invalid_penalty)

        if self.reward_type == "raw":

            reward = float(merge_score)

        elif self.reward_type == "log":

            reward = float(
                np.log1p(merge_score)
            )

        elif self.reward_type == "empty_cells":

            empty = np.count_nonzero(
                self.board == 0
            )

            reward = (
                float(merge_score)
                + self.empty_bonus * empty
            )

        elif self.reward_type == "corner":

            largest = np.max(self.board)

            corners = [
                self.board[0, 0],
                self.board[0, -1],
                self.board[-1, 0],
                self.board[-1, -1]
            ]

            reward = float(merge_score)

            if largest in corners:
                reward += self.corner_bonus

        elif self.reward_type == "combined":

            empty = np.count_nonzero(
                self.board == 0
            )

            largest = np.max(self.board)

            corners = [
                self.board[0, 0],
                self.board[0, -1],
                self.board[-1, 0],
                self.board[-1, -1]
            ]

            reward = (
                float(merge_score)
                + self.empty_bonus * empty
            )

            if largest in corners:
                reward += self.corner_bonus

        return float(reward)

    
    # RESET
       
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

    
     # STEP
       
    def step(self, action):

        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        self.total_steps += 1

        merge_score, valid_move = self._move(action)

        reward = self._get_reward(
            merge_score, valid_move
        )

        terminated = (
            self._is_game_over()
            or np.max(self.board) >= self.target_tile
        )

        # antes era siempre False y un movimiento invalido repetido no acababa nunca
        truncated = bool(
            self.total_steps >= self.max_episode_steps
            and not terminated
        )

        info = {
            "score": self.score,
            "steps": self.steps,
            "max_tile": int(np.max(self.board)),
            "empty_cells": int(
                np.count_nonzero(self.board == 0)
            ),
            "valid_move": valid_move
        }

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info
        )

    
    # RENDER
       
    def render(self):

        print("\n" + "=" * 30)

        print(self.board)

        print("=" * 30)

        print("Score:", self.score)
        print("Max tile:", np.max(self.board))
        print("Steps:", self.steps)

    def close(self):
        pass