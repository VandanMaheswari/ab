import os
import sys
import pandas as pd
from exception import CustomException
from utils import load_object




class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            
            model_path=os.path.join(r"model.pkl")
            preprocessor_path=os.path.join(r'preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,holiday,temp,clouds_all,weather_main,weekday,hour,month):

        self.holiday = holiday

        self.temp = temp

        self.clouds_all = clouds_all

        self.weather_main = weather_main

        self.weekday = weekday

        self.hour = hour

        self.month = month

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "holiday": [self.holiday],
                "temp": [self.temp],
                "clouds_all": [self.clouds_all],
                "weather_main": [self.weather_main],
                "weekday": [self.weekday],
                "hour": [self.hour],
                "month": [self.month],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
        
    
        
