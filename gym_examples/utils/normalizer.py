import pandas as pd
from typing import Tuple


class Normalizer():
    def __init__(self):
        self.std = None
        self.mean = None
        self.min = None
        self.max = None

    def norm_std(self, 
                 df_x: pd.DataFrame = None, 
                 df_y: pd.DataFrame | None = None,
                 ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        if df_x is not None: 
            self.mean = df_x.mean() 
            self.std = df_x.std()
            df_x = (df_x - self.mean) / self.std
        else:
            raise TypeError('Data must be pd.DataFrame format')
        if df_y is not None:
            col = df_y.columns
            df_y= (df_y - self.mean[col]) / self.std[col]
        
        return df_x, df_y
    
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
    
    def denorm_std(self,
                   df_x: pd.DataFrame | None = None, 
                   df_y: pd.DataFrame | None = None,
                   ) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:      
        if self.mean is not None and self.std is not None:     
            if df_x is not None:
                df_x= df_x * self.std + self.mean
            if df_y is not None:
                col = df_y.columns
                df_y = df_y * self.std[col] + self.mean[col]
        else:
            raise ValueError("No normalization parametrs found" )
                        
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
    
