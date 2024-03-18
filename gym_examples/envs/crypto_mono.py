from time import time
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

import gymnasium as gym


class Actions(Enum):
    Buy = 0
    Hold = 1
    Close = 2


class Positions(Enum):
    No_position = 0
    Long = 1

    def opposite(self):
        return Positions.Long if self == Positions.No_position else Positions.No_position


class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, 
                 df: pd.DataFrame, 
                 window_size: int,
                 features_names: list[str], 
                 frame_bound: tuple[int, int],
                 trade_fee: float,
                 amount: int,
                 #render_mode=None
                 ):
        
        assert df.ndim == 2
        #assert render_mode is None or render_mode in self.metadata['render_modes']

        #self.render_mode = render_mode

        self.df = df
        self.window_size = window_size
        self.features_names = features_names
        self.frame_bound = frame_bound
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        self.trade_fee = trade_fee
        self.amount = amount

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.shape, dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 2))))

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Positions.No_position
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info
    
    @property
    def _comission(self):
        return self.prices[self._current_tick] * self.amount * self.trade_fee
    
    @property
    def _hold_penalty(self):
        return self.prices[self._current_tick] * self.amount * 0.0001

    def step(self, action):
        self._truncated = False
        self._current_tick += 1
        step_reward = 0

        if self._current_tick == self._end_tick:
            self._truncated = True

        #если нет окрытых позиций
        if self._position == Positions.No_position:
            if action == Actions.Buy.value and not self._truncated:
                self._position = self._position.opposite()
                self._last_trade_tick = self._current_tick
                step_reward = 0
            elif action == Actions.Close.value or action == Actions.Hold.value:
                pass


        #если есть окрытые позиции
        elif self._position == Positions.Long:
            if action == Actions.Close.value or self._truncated:
                self._position = self._position.opposite()

                current_price = self.prices[self._current_tick]
                last_trade_price = self.prices[self._last_trade_tick]
                price_diff = current_price - last_trade_price
                duration = self._current_tick - self._last_trade_tick

                step_reward += (0.999 ** duration) * (price_diff * self.amount) - (2 * self._comission) #вычет комиссии

                shares = (self._total_profit * (1 - self.trade_fee)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee)) * current_price

            elif action == Actions.Hold.value or action == Actions.Buy.value:
                step_reward = 0 

        self._total_reward += step_reward

        #запись истории
        self._position_history.append(self._position)

        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.No_position:
                color = 'blue'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)

    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.No_position:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        normalizer = Normalizer()
        df = self.df[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        prices = df.loc[:,'close'].to_numpy()
        signal_features = df.loc[:,self.features_names]
        norm_signal_features, _ = normalizer.norm_minmax(df_x=signal_features, scale=(0,1))
        norm_signal_features = norm_signal_features.to_numpy()

        return prices.astype(np.float32), norm_signal_features.astype(np.float32)


class Normalizer():
    def __init__(self):
        self.min = None
        self.max = None
    
    def norm_minmax(self, 
                    df_x: pd.DataFrame = None, 
                    df_y: pd.DataFrame | None = None, 
                    scale: tuple[int, int] = (0, 1)
                    ) -> tuple[pd.DataFrame, pd.DataFrame | None]: 
        self.a, self.b = scale 
        if df_x is not None:
            self.min = df_x.min() 
            self.max = df_x.max() + 0.00001 #для избегания ошибки деления на ноль
            df_x = (self.b - self.a) * (df_x - self.min) / (self.max - self.min) + self.a
        else:
            raise TypeError('Data must be pd.DataFrame format')    
        if df_y is not None:
            col = df_y.columns
            df_y = (self.b - self.a) * (df_y - self.min[col]) / (self.max[col] - self.min[col]) + self.a  

        return df_x, df_y

    def denorm_minmax(self,
                      df_x: pd.DataFrame | None = None, 
                      df_y: pd.DataFrame | None = None,
                      ) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:         
        if self.min is not None and self.max is not None:
            if df_x is not None:
                df_x = (df_x - self.a) * (self.max-self.min) / (self.b - self.a) + self.min
            if df_y is not None:
                col = df_y.columns
                df_y = (df_y - self.a) * (self.max[col]-self.min[col]) / (self.b - self.a) + self.min[col]
        else:
            raise ValueError("No normalization parametrs found" )
        
        return df_x, df_y
    

