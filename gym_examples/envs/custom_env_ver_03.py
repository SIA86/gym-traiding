from time import time
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import gymnasium as gym


class Actions(Enum):
    Buy = 0
    Close_long = 1
    Sell = 2
    Close_short = 3
    Hold = 4
    Do_nothing = 5



class Positions(Enum):
    No_position = 0
    Long = 1
    Short = 2


class CryptoEnvQuantile_v3(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 3}
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 window_size: int,
                 frame_bound: Tuple[int, int],
                 features_names: list[str] = '',
                 price_type: str = 'Price_close',
                 trade_fee: float = 0.001,
                 initial_account: float = 100000.0,
                 max_loss: float = 0.25,
                 single_lot: int = 1000,
                 add_positions_info: bool = False,
                 render_mode=None,
                 **kwargs
                 ):
        
        assert dataframe.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.render_mode = render_mode
        self.add_positions_info = add_positions_info

        #параметры данных
        self.df = dataframe
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.features_names = features_names
        self.price_type = price_type

        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1] + int(self.add_positions_info))

        #параметры портфеля
        self.trade_fee = trade_fee
        self.single_lot = single_lot
        self.initial_account = initial_account
        self.max_loss = max_loss

        self.start_tick = self.window_size
        self.end_tick = len(self.prices) - 1

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-1000, high=1000, shape=self.shape, dtype=np.float32,
        )

        #парметры эпизода
        self.truncated = None #флаг на случай если депозит уходит в ноль или ниже разрешенного уровня
        self.done = None #флаг сигнализирующий о конце датасета

        self.current_tick = None #номер текущей свечи в последовательности
        self.last_buy_tick = None #цена последней сделки long
        self.last_sell_tick = None #цена последней сделки short
        self.position = None #флаг сигнализируещеей о наличии открытой позиции
        self.position_history = None #журнал сделок
        self.coins = None #кол-во лотов в портфеле
    
        self.cash = None #свободные средства
        self.account = None #депозит
        self.total_reward = None #итоговая награда
        self.total_profit = None #прибыль/убыток

        self.first_rendering = None #параметры отрисовки
        self.history = None #журнал общий

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 2))))

        self.truncated = False #сброс флага 
        self.done = False  #сброс флага 

        self.current_tick = self.start_tick #текущая цена равна стартовой цене
        self.last_buy_tick = None #цена последней сделки long
        self.last_sell_tick = None #цена последней сделки short
        self.position = Positions.No_position #сброс состояния портфеля на "нет позиций"
        self.position_history = [self.position.value for _ in range(self.window_size + 1)] #запись в историю сделок "нет позиции" длиной ширина окна

        self.coins = 0 #начальное кол-во монет
        current_price = self.prices[self.current_tick] #цена монеты на момент старта
        self.cash = self.initial_account  #стартовый кэш
        self.account = self.cash + current_price * self.coins #депозит как сумма кэша и акций, выраженных в цене

        self.total_reward = 0
        self.total_profit = 0
        
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info(None)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self.current_tick += 1
        step_reward = 0
        step_bonus_rew = 0
        step_penalty = 0
        current_price = self.prices[self.current_tick]

        if self.current_tick == self.end_tick: #проверка на конец датасета
            self.done = True

        if self.account <= self.initial_account * (1 - self.max_loss): #проверка состояния депозита
            self.truncated = True #если просадка больше self.loss %, то торговля заканчивается
 
        if all([self.position == Positions.No_position, #если нет открытых позиций
                not self.done,
                not self.truncated]): 
            self.coins += int(self.single_lot / current_price)
            if action == Actions.Buy.value: #если НС предсказывает покупать
                self.position = Positions.Long 
                self.cash -= current_price * self.coins * (1 + self.trade_fee)
                self.last_buy_tick = self.current_tick #сохраняем индекс свечи, когда была покупка
            
            elif action == Actions.Sell.value: #если НС предсказывает покупать
                self.position = Positions.Short
                self.cash -= current_price * self.coins * (1 + self.trade_fee)
                self.last_sell_tick = self.current_tick #сохраняем индекс свечи, когда была продажа

            elif action == Actions.Close_long.value: #если НС предсказывает покупать
                pass
            elif action == Actions.Close_short.value:
                pass
            elif action == Actions.Hold.value:
                pass
            elif action == Actions.Do_nothing.value:
                pass

        elif self.position == Positions.Long: #если есть long positions
            if action == Actions.Buy.value: #если НС предсказывает покупать
                pass
            elif action == Actions.Sell.value: #если НС предсказывает покупать
                pass           
            elif action == Actions.Close_long.value: #если НС предсказывает покупать
                buy_price = self.prices[self.last_buy_tick]
                comission = (current_price + buy_price) * self.trade_fee * self.coins

                step_reward += (current_price - buy_price) * self.coins - comission

            elif action == Actions.Close_short.value:
                pass
            elif action == Actions.Hold.value:
                pass
            elif action == Actions.Do_nothing.value:
                pass

        elif self.position == Positions.Short: #если есть окрытые позиции
            if action == Actions.Buy.value: #если НС предсказывает покупать
                pass
            elif action == Actions.Sell.value: #если НС предсказывает покупать
                pass    
            elif action == Actions.Close_long.value: #если НС предсказывает покупать
                pass
            elif action == Actions.Close_short.value:
                sell_price = self.prices[self.last_sell_tick]
                comission = (current_price + sell_price) * self.trade_fee * self.coins

                step_reward += (sell_price - current_price) * self.coins - comission

            elif action == Actions.Hold.value:
                pass
            elif action == Actions.Do_nothing.value:
                pass
       
        next_account = self.cash + current_price * self.coins #вычисление состояния текущего портфеля
        step_reward += (next_account - self.account) + step_bonus_rew + step_penalty #расчет вознаграждения, как величина изменения портфеля
        self.account = next_account #перезапись состояния портфеля на новый

        if any([self.done, self.truncated]): #если конец
            self.total_profit = self.account / self.initial_account #итоговый доход

        #запись истории
        self.total_reward += step_reward
        self.position_history.append(self.position.value)

        observation = self._get_observation()
        info = self._get_info(action)
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self.done, info

    def _get_info(self, action):
        return dict(
            total_reward=self.total_reward,
            total_profit=self.total_profit,
            position=self.position.value,
            actions=action
        )

    def _get_observation(self):
        obs = self.signal_features[(self.current_tick-self.window_size):self.current_tick]
        if self.add_positions_info:
            position_column = np.asarray(self.position_history[(self.current_tick-self.window_size):self.current_tick])[:,np.newaxis]
            obs = np.concatenate([obs, position_column], axis = 1)
        return obs.astype(np.float32)

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
                color = 'gray'
            elif position == Positions.Long:
                color = 'green'
            elif position == Positions.Short:
                color = 'red'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self.position_history[self.start_tick]
            _plot_position(start_position, self.start_tick)

        _plot_position(self.position, self.current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)

    def render_all(self, title=None):
        window_ticks = np.arange(len(self.position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        no_position_ticks = []

        for i, tick in enumerate(window_ticks):
            if self.position_history[i] == Positions.No_position.value:
                no_position_ticks.append(tick)
            elif self.position_history[i] == Positions.Long.value:
                long_ticks.append(tick)
            elif self.position_history[i] == Positions.Short.value:
                short_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'red')
        plt.plot(long_ticks, self.prices[long_ticks], 'green')
        plt.plot(no_position_ticks, self.prices[no_position_ticks], 'gray')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
        df = self.df[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        prices = df.loc[:,self.price_type].to_numpy()
        signal_features = df.loc[:,self.features_names].to_numpy()
        signal_features = quantile_transformer.fit_transform(signal_features)

        return prices.astype(np.float32), signal_features.astype(np.float32)
    

class CryptoEnvMinMaxScaler_v3(CryptoEnvQuantile_v3):
    def _process_data(self):
        scaler = MinMaxScaler()
        df = self.df[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        prices = df.loc[:,self.price_type].to_numpy()
        signal_features = df.loc[:,self.features_names].to_numpy()
        signal_features = scaler.fit_transform(signal_features)

        return prices.astype(np.float32), signal_features.astype(np.float32)