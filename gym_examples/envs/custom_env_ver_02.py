from time import time
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym

import plotly.graph_objects as go


class Actions(Enum):
    Buy = 1
    Close_long = 2
    Sell = 3
    Close_short = 4
    Hold = 0


class Positions(Enum):
    No_position = 0
    Long = 0.5
    Short = 1


class CryptoEnvMinMaxScaler_v2(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 3}
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 window_size: int,
                 frame_bound: Tuple[int, int],
                 features_names: list[str] = '',
                 price_type: str = 'Price_close',
                 trade_fee: float = 0.001,
                 max_hold_duration: int = 600,
                 add_positions_info: bool = False,
                 render_mode=None,
                 **kwargs
                 ):
        
        assert dataframe.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.render_mode = render_mode
        self.add_positions_info = add_positions_info

        #параметры данных
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.features_names = features_names
        self.price_type = price_type
        self.df = dataframe[self.frame_bound[0]-self.window_size:self.frame_bound[1]].reset_index(drop=True)

        self.prices, self.norm_prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1] + int(self.add_positions_info))

        #параметры портфеля
        self.trade_fee = trade_fee
        self.max_hold_duration = max_hold_duration

        self.start_tick = self.window_size
        self.end_tick = len(self.prices) - 1

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.shape, dtype=np.float32,
        )

        #парметры эпизода
        self.done = None #флаг сигнализирующий о конце датасета
        self.current_tick = None #номер текущей свечи в последовательности
        self.last_buy_tick = None #цена последней сделки long
        self.last_sell_tick = None #цена последней сделки short
        self.position = None #флаг сигнализируещеей о наличии открытой позиции
        self.position_history = None #журнал сделок
        self.coins = None #кол-во лотов в портфеле
        self.hold_duration = None
    
        self.total_reward = None
        self.total_profit = None
        self.running_profit = None
        self.short_num = None
        self.short_profit = None
        self.long_num = None
        self.long_profit = None
        self.durations = None
        self.first_rendering = None #параметры отрисовки
        self.history = None #журнал общий
        self.trades = None #журнал сделок

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 100))))

        self.done = False  #сброс флага 
        self.current_tick = self.start_tick #текущая цена равна стартовой цене
        self.last_buy_tick = None #цена последней сделки long
        self.last_sell_tick = None #цена последней сделки short
        self.hold_duration = 0
        self.position = Positions.No_position #сброс состояния портфеля на "нет позиций"
        self.position_history = [self.position.value for _ in range(self.window_size)] #запись в историю сделок "нет позиции" длиной ширина окна

        self.total_reward = 0
        self.running_profit = 1
        self.total_profit = 1
        self.short_num = 0
        self.short_profit = 0
        self.long_num = 0
        self.long_profit = 0
        self.durations = []
        self.trades = pd.DataFrame(columns=['start_time', 'type_dir'])

        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info(np.NaN)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self.current_tick += 1
        step_reward = 0
        step_penalty = 0
        current_price = self.norm_prices[self.current_tick]

        if self.current_tick == self.end_tick: #провеmax_lossрка на конец датасета
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
            
            elif action == Actions.Sell.value: #если НС предсказывает покупать
                self.position = Positions.Short #меняем позицию на шорт
                self.hold_duration = 0  #сбрасываем счетчик длительности
                self.last_sell_tick = self.current_tick #сохраняем индекс свечи, когда была продажа

                self.trade = pd.DataFrame({
                    'start_time':[self.df.loc[self.current_tick, 'datetime_close']],
                    'type_dir':[self.position.value]
                })
                self.trades = pd.concat([self.trades, self.trade], axis=0) 

            elif action == Actions.Close_long.value:
                self.hold_duration += 1
                step_penalty += current_price * 0.01
            elif action == Actions.Close_short.value:
                self.hold_duration += 1
                step_penalty += current_price * 0.01
            elif action == Actions.Hold.value:
                self.hold_duration += 1 
          
        elif self.position == Positions.Long: #если есть long positions        
            if any([action == Actions.Close_long.value,                               
                    self.done]): #если НС предсказывает закрыть лонг
                self.position = Positions.No_position #меняем позицию на нет позиций

                buy_price = self.norm_prices[self.last_buy_tick] #определяем цену входа в лонг по индексу
                comission = (current_price + buy_price) * self.trade_fee #расчет комиссии

                profit_percents = abs(current_price / buy_price - 1)
                reward = (current_price - buy_price) - comission

                temp_profit = self.running_profit / (current_price * (1 + self.trade_fee))
                self.running_profit = temp_profit * (current_price * (1 - self.trade_fee))

                if profit_percents > 0.02:
                    step_reward += 2 * reward 
                elif profit_percents <= 0.02 and profit_percents >= 0.01: 
                    step_reward += reward
                else:
                    step_reward += 0.5 * reward

                self.long_num += 1
                self.long_profit += (self.running_profit - self.total_profit)
                self.total_profit = self.running_profit

                self.durations.append(self.hold_duration)  
                self.hold_duration = 0 #сбрасываем счетчик удержания

                self.trade = pd.DataFrame({
                    'start_time':[self.df.loc[self.current_tick, 'datetime_close']],
                    'type_dir':[self.position.value]
                })
                self.trades = pd.concat([self.trades, self.trade], axis=0) 


            elif action == Actions.Buy.value: #если НС предсказывает покупать
                step_penalty += current_price * 0.01
                self.hold_duration += 1 #считаем длительность 
            elif action == Actions.Sell.value: #если НС предсказывает продавать
                step_penalty += current_price * 0.01
                self.hold_duration += 1 #считаем длительность  
            elif action == Actions.Close_short.value: #если НС предсказывает закрыть шорт
                step_penalty += current_price * 0.01
                self.hold_duration += 1 #считаем длительность 
            elif action == Actions.Hold.value: #если НС предсказывает удерживать
                self.hold_duration += 1 #считаем время удержания
            
        elif self.position == Positions.Short: #если есть окрытые позиции
            if any([action == Actions.Close_short.value,  
                    self.done]): #если НС предсказывает закрыть шорт
                self.position = Positions.No_position #меняем позиуии на нет позиций

                sell_price = self.prices[self.last_sell_tick] #определяем цену входа в шорт по индексу
                comission = (current_price + sell_price) * self.trade_fee #расчет комиссии

                profit_percents = abs(sell_price / current_price - 1)
                reward = (sell_price - current_price) - comission

                temp_profit = self.running_profit * (current_price * (1 - self.trade_fee))
                self.running_profit = temp_profit / (current_price * (1 + self.trade_fee))

                if profit_percents > 0.02:
                    step_reward += 2* reward
                elif profit_percents <= 0.02 and profit_percents >= 0.01: 
                    step_reward += reward
                else:
                    step_reward += 0.5 * reward

                self.short_num += 1
                self.short_profit += self.running_profit - self.total_profit
                self.total_profit = self.running_profit

                self.durations.append(self.hold_duration)    

                self.hold_duration = 0 #сбрасываем счетчик удержания

                self.trade = pd.DataFrame({
                    'start_time':[self.df.loc[self.current_tick, 'datetime_close']],
                    'type_dir':[self.position.value]
                })
                self.trades = pd.concat([self.trades, self.trade], axis=0) 

            elif action == Actions.Buy.value: #если НС предсказывает покупать
                step_penalty += current_price * 0.01
                self.hold_duration += 1 #считаем длительность 
            elif action == Actions.Sell.value: #если НС предсказывает покупать
                step_penalty += current_price * 0.01
                self.hold_duration += 1 #считаем длительность   
            elif action == Actions.Close_long.value: #если НС предсказывает покупать
                step_penalty += current_price * 0.01
                self.hold_duration += 1 #считаем длительность     
            elif action == Actions.Hold.value: #если НС предсказывает удерживать
                self.hold_duration += 1 #считаем время удержания

        if self.hold_duration >= self.max_hold_duration: #если удержание дольше максимального
            step_penalty += current_price * 0.01 #расчет штрафа

        step_reward -=  step_penalty
            
        #запись истории
        self.total_reward += step_reward
        self.position_history.append(self.position.value)

        observation = self._get_observation()
        info = self._get_info(int(action))
        self._update_history(info)

        if self.done:
            self.trades = self.trades.reset_index(drop=True)
            
            print("\n".join([
                f"Total profit: {info['total_profit']:.4f}",
                f"Longs num: {info['long_num']}, profit: {info['long_profit']:.2f}",
                f"Short num: {info['short_num']}, profit: {info['short_profit']:.2f}",
                f"Mean duration: {np.mean(info['duration']):.2f}"
                ])
            )
            
            self.render_all()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self.done, info

    def _get_info(self, action):
        return dict(
            total_reward=self.total_reward,
            total_profit=self.total_profit,
            long_num = self.long_num,
            short_num = self.short_num,
            long_profit = self.long_profit,
            short_profit = self.short_profit,
            duration = self.durations,
            position=self.position.value,
            action=action,
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
        fig, (pl1, pl2) = plt.subplots(2,1, sharex=True, figsize=(12,8))
        pl1.plot(self.prices)

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

        pl1.plot(short_ticks, self.prices[short_ticks], 'ro')
        pl1.plot(long_ticks, self.prices[long_ticks], 'go')
        pl1.plot(no_position_ticks, self.prices[no_position_ticks], 'bo')
        pl1.set_title("Green - long position, Red - short position, Blue - no position")
        pl1.grid(True)

        actions = [np.NaN for _ in range(self.window_size)] + self.history['action']
        pl2.plot(actions)
        pl2.grid(True)
        pl2.set_title("Actions: Buy-0, Close_buy-1, Sell-2, Close_sell-3, Hold-4")

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.total_profit
        )
        plt.show()

    def plotly_all(self):
        fig = go.Figure(data=[go.Candlestick(
                x=self.df['datetime_close'],
                open=self.df['Price_open'],
                high=self.df['Price_high'],
                low=self.df['Price_low'],
                close=self.df['Price_close'],
                text=self.df.index )])

        for index, row in self.trades.loc[self.trades['type_dir'] ==  1].iterrows():
            fig.add_vline(x=self.trades['start_time'].loc[index], line_width=3, line_dash="dash", line_color="green")

        for index, row in self.trades.loc[(self.trades['type_dir'] ==  2)].iterrows():
            fig.add_vline(x=self.trades['start_time'].loc[index], line_width=3, line_dash="dash", line_color="red")

        for index, row in self.trades.loc[(self.trades['type_dir'] ==  0)].iterrows():
            fig.add_vline(x=self.trades['start_time'].loc[index], line_width=3, line_dash="dash", line_color="gray")

        fig.update_layout(
            autosize=True,
        )

        fig.show()

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        scaler = MinMaxScaler(feature_range=(0.00001, 1))
        prices = self.df.loc[:,self.price_type].to_numpy()
        signal_features = self.df.loc[:,self.features_names]
        signal_features = scaler.fit_transform(signal_features)
        indx = scaler.feature_names_in_.tolist().index(self.price_type)
        norm_prices = signal_features[:,indx]

        return prices.astype(np.float32), norm_prices.astype(np.float32), signal_features.astype(np.float32)
    
