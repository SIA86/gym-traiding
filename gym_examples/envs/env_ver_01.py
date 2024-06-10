"""
Торговля: только лонг
Действия: купить / закрыть покупку / удерживать
Награда: прибыль/убыток от закрытия позиций
Доп.штрафы: за бессмысленные действия
"""

from time import time
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from IPython.display import display, clear_output
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


class Actions(Enum):
    Buy = 0
    Hold = 1
    Close = 2


class Positions(Enum):
    No_position = 0
    Long = 1

    def opposite(self):
        return Positions.Long if self == Positions.No_position else Positions.No_position


class EnvTrain(gym.Env):
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 frame_bound: tuple[int, int],
                 window_size: int = 60,
                 features_names: list[str] = '',
                 episode_length: int = 1000,
                 trade_fee: float = 0.001,
                 penalty_mult: float = 0.01,
                 ):
        assert dataframe.ndim == 2

        self.frame_bound = frame_bound
        self.window_size = window_size
        self.episode_length = episode_length
        self.features_names = features_names
        self.trade_fee = trade_fee
        self.penalty_mult = penalty_mult

        self.df = dataframe[self.frame_bound[0]-self.window_size:self.frame_bound[1]].reset_index(drop=True)
        self.prices, self.norm_prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(
            low=self.signal_features.min(), high=self.signal_features.max(), shape=self.shape, dtype=np.float32,
        )

        #парметры эпизода
        self.done = None #флаг сигнализирующий о конце датасета
        self.start_tick = None
        self.current_tick = None #номер текущей свечи в последовательности
        self.end_tick = None
        self.last_buy_tick = None #цена последней сделки long
        self.position = None #флаг сигнализируещеей о наличии открытой позиции
        self.position_history = None #журнал сделок
        self.total_reward = None
        self.total_profit = None
        self.cummulative_total_reward = None
        self.cummulative_total_profit = None
        self.deals = None
        self.history = None #журнал общий
        self.trades = None #журнал сделок
        self.hold_duration = None
        self.episode = 0
        self.all_total_rewards = []
        self.all_total_profits = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 100))))
        self.start_tick = random.sample(range(self.window_size, len(self.prices) - self.episode_length), 1)[0]#текущая цена равна стартовой цене
        self.end_tick = self.start_tick + self.episode_length
        self.current_tick = self.start_tick
        self.done = False  #сброс флага 
        self.last_buy_tick = None #цена последней сделки long
        self.last_sell_tick = None #цена последней сделки short
        self.position = Positions.No_position #сброс состояния портфеля на "нет позиций"
        self.position_history = [] #запись в историю сделок "нет позиции" длиной ширина окна
        self.episode += 1

        self.total_reward = 0
        self.total_profit = 1
        self.cummulative_total_reward = []
        self.cummulative_total_profit = []
        self.hold_duration = 0

        self.durations = []
        self.deals = 0
        self.trades = pd.DataFrame(columns=['start_time', 'type_dir'], dtype=(int, int))
        self.history = {}

        observation = self._get_observation()
        info = self._get_info(np.NaN)

        return observation, info
    
    def step(self, action):
        step_penalty = 0
        step_reward = 0
        self.current_tick += 1

        current_price = self.prices[self.current_tick]
        current_norm_price = self.norm_prices[self.current_tick]

        if self.current_tick == self.end_tick:
            self.done = True
 
        if all([self.position == Positions.No_position, #если нет открытых позиций
                not self.done]): 

            if action == Actions.Buy.value: #если НС предсказывает покупать
                self.position = Positions.Long #меняем позицию на лонг
                self.hold_duration = 0 #сбрасываем счетчик длительности
                self.last_buy_tick = self.current_tick #сохраняем индекс свечи, когда была покупка

                self.trade = pd.DataFrame({
                    'start_time':[self.df.loc[self.current_tick, 'datetime_close']],
                    'type_dir':[self.position.value]
                })
                self.trades = pd.concat([self.trades, self.trade], axis=0) 
            
            elif action == Actions.Close.value: #если НС предсказывает покупать
                self.hold_duration += 1  #сбрасываем счетчик длительности
                step_penalty = current_norm_price * self.penalty_mult

            elif action == Actions.Hold.value:
                self.hold_duration += 1 
          
        elif self.position == Positions.Long: #если есть long positions        
            if any([action == Actions.Close.value,                               
                    self.done]): #если НС предсказывает закрыть лонг
                self.position = Positions.No_position #меняем позицию на нет позиций

                buy_price = self.prices[self.last_buy_tick]
                buy_norm_price = self.norm_prices[self.last_buy_tick] #определяем цену входа в лонг по индексу
                comission = (current_norm_price + buy_norm_price) * self.trade_fee #расчет комиссии

                step_reward = (current_norm_price - buy_norm_price) - comission

                temp_profit = self.total_profit  / (buy_price * (1 + self.trade_fee))
                self.total_profit = temp_profit * (current_price * (1 - self.trade_fee))

                self.deals += 1

                self.durations.append(self.hold_duration)  
                self.hold_duration = 0 #сбрасываем счетчик удержания

                self.trade = pd.DataFrame({
                    'start_time':[self.df.loc[self.current_tick, 'datetime_close']],
                    'type_dir':[self.position.value]
                })
                self.trades = pd.concat([self.trades, self.trade], axis=0) 


            elif action == Actions.Buy.value: #если НС предсказывает покупать
                step_penalty = current_norm_price * self.penalty_mult
                self.hold_duration += 1 #считаем длительность 
    
            elif action == Actions.Hold.value: #если НС предсказывает удерживать
                self.hold_duration += 1 #считаем время удержания

        step_reward -=  step_penalty
            
        #запись истории
        self.total_reward += step_reward
        self.position_history.append(self.position.value)
        self.cummulative_total_reward.append(self.total_reward)
        self.cummulative_total_profit.append(self.total_profit)

        observation = self._get_observation()
        info = self._get_info(int(action))
        self._update_history(info)
            
        if self.done:
            self.trades = self.trades.reset_index(drop=True)
            self.all_total_rewards.append(self.total_reward)
            self.all_total_profits.append(self.total_profit)

            self.plot_results(info)

        return observation, step_reward, False, self.done, info

    def _get_info(self, action):
        return dict(
            total_reward=self.total_reward,
            total_profit=self.total_profit,
            deals = self.deals,
            duration = self.durations,
            position=self.position.value,
            cum_reward = self.cummulative_total_reward,
            cum_profit = self.cummulative_total_profit,
            action=action,
        )

    def _get_observation(self):
        obs = self.signal_features[(self.current_tick-self.window_size):self.current_tick]
        return obs.astype(np.float32)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def plot_results(self, info):
        #вывод графиков
        fig, self.axes = plt.subplots(6,1, figsize=(12,16), sharex=True, gridspec_kw={'height_ratios': [3,3,2,2,4,2]})
        fig.suptitle(f"Metrics at episode-{self.episode}")
        local_prices = self.prices[self.start_tick:self.end_tick]
        window_ticks = np.arange(len(self.position_history))

        self.axes[0].plot(self.all_total_rewards)
        self.axes[0].set_title(f"Total reward: {info['total_reward']:.2f}")
        self.axes[0].grid(True)

        self.axes[1].plot(self.all_total_profits)
        self.axes[1].set_title(f"Total profit: {info['total_profit']:.4f}")
        self.axes[1].grid(True)

        self.axes[2].plot(info['cum_reward'])
        self.axes[2].set_title("Cummulative reward (CR)")
        self.axes[2].grid(True)

        self.axes[3].plot(info['cum_profit'])
        self.axes[3].set_title("Cummulative profit (CP)")
        self.axes[3].grid(True)

        self.axes[4].plot(local_prices)

        long_ticks = []
        no_position_ticks = []

        for i, tick in enumerate(window_ticks):
            if self.position_history[i] == Positions.No_position.value:
                no_position_ticks.append(tick)
            elif self.position_history[i] == Positions.Long.value:
                long_ticks.append(tick)
  
        self.axes[4].plot(long_ticks, local_prices[long_ticks], 'go')
        self.axes[4].plot(no_position_ticks, local_prices[no_position_ticks], 'bo')
        self.axes[4].set_title(f"Episode length: {self.episode_length}, num deals: {info['deals']}, mean duration: {np.mean(info['duration']):.2f}")
        self.axes[4].grid(True)

        actions = self.history['action']
        self.axes[5].plot(actions)
        self.axes[5].grid(True)
        self.axes[5].set_yticks(np.arange(3), ['Buy', 'Hold', 'Close']) 
        buys, holds, closes = map(actions.count, [0,1,2])
        self.axes[5].set_title(f"Buys-{buys}, Holds-{holds}, Closes-{closes}")

        display(fig)    
        clear_output(wait=True)

    def _process_data(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        norm_algorythm = QuantileTransformer(output_distribution='normal', random_state=0) 

        prices = self.df.loc[:,'Price_close'].to_numpy()
        signal_features = self.df.loc[:,self.features_names]
        signal_features = norm_algorythm.fit_transform(signal_features)
        signal_features = scaler.fit_transform(signal_features)
        indx = norm_algorythm.feature_names_in_.tolist().index('Price_close')
        norm_prices = signal_features[:,indx]

        return prices.astype(np.float32), norm_prices.astype(np.float32), signal_features.astype(np.float32)
    

class EnvVal(EnvTrain):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 100))))

        self.start_tick = self.window_size #текущая цена равна стартовой цене
        self.current_tick = self.start_tick
        self.end_tick = len(self.prices) - 1 

        self.done = False  #сброс флага 
        self.last_buy_tick = None #цена последней сделки long
        self.last_sell_tick = None #цена последней сделки short
        self.position = Positions.No_position #сброс состояния портфеля на "нет позиций"
        self.position_history = [] #запись в историю сделок "нет позиции" длиной ширина окна

        self.total_reward = 0
        self.total_profit = 1
        self.cummulative_total_reward = []
        self.cummulative_total_profit = []
        self.hold_duration = 0

        self.durations = []
        self.deals = 0
        self.trades = pd.DataFrame(columns=['start_time', 'type_dir'], dtype=(int, int))
        self.history = {}

        observation = self._get_observation()
        info = self._get_info(np.NaN)

        return observation, info