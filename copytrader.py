import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt
import array as arr

import tensorflow as tf

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4092)])
#tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)




from datetime import datetime
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import numpy as np
from tensorflow.keras import models
import time
import random

import plotly.express as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import Input 
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import model_from_json

import tensorflow as tf
from tensorflow import keras



class Get_vectors:
    def __init__(self, rates_frame):
        self.rates_frame_op = rates_frame
    #+---------------------------------------------------------------------------------
    def range_open_close(self):
        cum=[]
        for ip in range(len(self.rates_frame_op['open'])):
            cum.append(abs(self.rates_frame_op['open'][ip]-self.rates_frame_op['close'][ip]))
        return cum
    #+---------------------------------------------------------------------------------
    def range_high_low(self):
        cum=[]
        for ip in range(len(self.rates_frame_op['high'])):
            cum.append(abs(self.rates_frame_op['high'][ip]-self.rates_frame_op['low'][ip]))
        return cum
    #+---------------------------------------------------------------------------------
    def range_high(self):
        cum=[]
        for ip in range(len(self.rates_frame_op['high'])):
            cum.append(abs(self.rates_frame_op['high'][ip]))
        return cum
    #+---------------------------------------------------------------------------------
    def range_low(self):
        cum=[]
        for ip in range(len(self.rates_frame_op['low'])):
            cum.append(abs(self.rates_frame_op['low'][ip]))
        return cum
    #+---------------------------------------------------------------------------------
    def range_tick(self):
        cum=[]
        for ip in range(len(self.rates_frame_op['tick_volume'])):
            cum.append(abs(self.rates_frame_op['tick_volume'][ip]))
        return cum


class Get_connection:
    #+---------------------------------------------------------------------------------
    def __init__(self, login, server , password):
        self.login = login
        self.server = server
        self.password = password
    #+---------------------------------------------------------------------------------    
    def Get_conn(self):
        # display data on the MetaTrader 5 package
        print("MetaTrader5 package author: ",mt5.__author__)
        print("MetaTrader5 package version: ",mt5.__version__)
        
        # import the 'pandas' module for displaying data obtained in the tabular form 
        pd.set_option('display.max_columns', 500) # number of columns to be displayed
        pd.set_option('display.width', 1500)      # max table width to display
        # import pytz module for working with time zone
        
        #if not mt5.initialize(login=64692122, server="XMGlobal-MT5 2",password="Cenidet2015"):
        if not mt5.initialize(login=self.login, server=self.server,password=self.password):
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        # establish connection to MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
    #+---------------------------------------------------------------------------------
    def Mt5_down(self):
        mt5.shutdown()


class Get_data:
    #+---------------------------------------------------------------------------------
    def __init__(self, year, month, day, currency, tmframe, numsamples):
        self.year=year
        self.month=month 
        self.day=day 
        self.currency=currency
        self.tmframe=tmframe
        self.numsamples=numsamples
        self.timezone = pytz.timezone("Etc/UTC")
    #+---------------------------------------------------------------------------------
    def get_rates(self):#year, month, day, currency, tmframe, numsamples):
        utc_from = datetime(self.year, self.month, self.day, tzinfo=self.timezone)
        rates = mt5.copy_rates_from(self.currency, self.tmframe, utc_from, self.numsamples)

        #print(rates)
        return rates

###### recibe los datos obtenidos y la clase de extraccion de vectores
class Get_statistic():
    #+---------------------------------------------------------------------------------
    def __init__(self, rates, name):
        self.rates=rates
        self.name=name
    #+---------------------------------------------------------------------------------
    # set time zone to UTC
    # Example statistical_coins(2018, 1, 10, "EURUSD", mt5.TIMEFRAME_H4, 10000)
    def rates_toratesframe(self):
        #print("Display obtained data 'as is'")
        #for rate in self.rates:
        #    print(rate)
        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(self.rates)
        # convert time in seconds into the datetime format
        rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')         
        # display data
        print("\nDisplay dataframe with data")
        print(rates_frame)
        return rates_frame

    #+---------------------------------------------------------------------------------
    def statistical_coins(self,rates_toratesframe):
        #rates = self.rates
        # shut down connection to the MetaTrader 5 terminal
        # display each element of obtained data in a new line
        
        # tick volume son todos los cambios ocurridos en esa barra
        # time     open     high      low    close  tick_volume  spread  real_volume
        #numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)
        #+---------------------------------------------------------------------------------
        gvec=Get_vectors(rates_toratesframe)

        cumul_openclose=gvec.range_high_low()
        #print(cumul_openclose)

        cumul_higlow=gvec.range_open_close()
        #print(cumul_higlow)

        cumul_hig=gvec.range_high()
        #print(cumul_hig)

        cumul_low=gvec.range_low()
        #print(cumul_low)

        cumul_tick=gvec.range_tick()
        #print(cumul_tick)

        hhoc,bioc=np.histogram(cumul_openclose, bins=256, range=None, normed=None, weights=None, density=False)
        hhhl,bihl=np.histogram(cumul_higlow, bins=256, range=None, normed=None, weights=None, density=False)
        hhh,bih=np.histogram(cumul_hig, bins=256, range=None, normed=None, weights=None, density=False)
        hhl,bil=np.histogram(cumul_low, bins=256, range=None, normed=None, weights=None, density=False)
        htk,btk=np.histogram(cumul_tick, bins=256, range=None, normed=None, weights=None, density=False)    ################ cumulo de volumenes de transacciones 

        plt.figure(figsize=(25, 5))

        plt.subplot(151)
        #plt.hist(cumul_openclose, bins='auto')
        plt.plot(np.array(bioc[0:len(bioc)-1]),np.array(hhoc))
        plt.xlabel('open-close_widht___'+self.name)
        plt.ylabel('samples')

        plt.subplot(152)
        #plt.hist(cumul_higlow, bins='auto')
        plt.plot(np.array(bihl[0:len(bihl)-1]),np.array(hhhl))
        plt.xlabel('high-low_widht___'+self.name)
        plt.ylabel('samples')

        plt.subplot(153)
        #plt.hist(cumul_hig, bins='auto')
        plt.plot(np.array(bih[0:len(bih)-1]),np.array(hhh))
        plt.xlabel('high___'+self.name)
        plt.ylabel('samples')

        plt.subplot(154)
        #plt.hist(cumul_low, bins='auto')
        plt.plot(np.array(bil[0:len(bil)-1]),np.array(hhl))
        plt.xlabel('low___'+self.name)
        plt.ylabel('samples')

        plt.subplot(155)
        plt.plot(np.array(btk[0:len(btk)-1]),np.array(htk))
        plt.xlabel('tick___'+self.name)
        plt.ylabel('samples')
        plt.show()
        return


class Vector_to_train():
    def __init__(self, rates, name, window, whatlearn):             #############################################
        self.rates=Get_statistic(rates,name)            ######## information 
        self.proc_rates=self.rates.rates_toratesframe() ########
        self.whatlearn = whatlearn
        ######################################################## en este lugar definir un frup de palabras para identificar que parte de la informacion se va  a tomar para el entrenamiento
        if self.whatlearn == 'val_h':
            self.cumul_hig=Get_vectors(self.proc_rates).range_high()
        if self.whatlearn == 'cumul':
            self.cumul_hig=Get_vectors(self.proc_rates).range_tick()     ######## tengo el vector de acumulados altos, en este lugar se cambia al tipo de datos (ej: range_tick) what to train 


        self.window=window
#+----------------------------------------------------------------------------
    def get_h(self):
        maxval=max(self.cumul_hig)
        minval=min(self.cumul_hig)
        Hparam=(maxval-minval)/20
        return Hparam
#+---------------------------------------------------------------------------
    def get_sum(self):
        posi=0
        acum=[]
        for ip in range(int(len(self.cumul_hig)/self.window)-1):
            acum.append(sum(self.cumul_hig[posi:posi+self.window]))
            posi+=(self.window)
        return acum
#+-----------------------------------------------------------------------------
    def concat(far,sar):
        arr1 = np.array(far)
        arr2 = np.array(sar)
        arr = np.concatenate((arr1, arr2))
        return arr

    def create_arrays(self):
        sell_0=[]
        down_1=[]
        buy_2=[]
        up_3=[]
        for ip in range(self.window):
            sell_0.append(0)
            down_1.append(1)
            buy_2.append(2)
            up_3.append(3)
        total=[]
        total.append(sell_0)
        total.append(down_1)
        total.append(buy_2)
        total.append(up_3)
        return total


    def get_train_vector(self, get_sum, get_h, create_arrays):
        vsuma=get_sum(self)
        train_vec=[]
        vset=create_arrays(self)

        for ip in range(1,len(vsuma)-1):
            if vsuma[ip-1]>(vsuma[ip]+get_h(self)) and vsuma[ip]>(vsuma[ip+1]+get_h(self)):
                train_vec+=vset[0]
            if vsuma[ip-1]>(vsuma[ip]+get_h(self)) and (vsuma[ip]+get_h(self))<(vsuma[ip+1]):
                train_vec+=vset[1]
            if (vsuma[ip-1]+get_h(self))<vsuma[ip] and (vsuma[ip]+get_h(self))<vsuma[ip+1]:
                train_vec+=vset[2]
            if (vsuma[ip-1]+get_h(self))<vsuma[ip] and vsuma[ip]>(vsuma[ip+1]+get_h(self)):
                train_vec+=vset[3]
        return train_vec
       
class Data_learning():
    def __init__(self, name_coin, timep, final_vec, split, look_back, num_epoch, losse, met, whatlearn):  ## NZDJPY, H1, final_vec ,split=0.8, 10, 1000 ,'mean_squared_error', 'accuracy'
        self.final_vec=final_vec
        self.split_percent = split
        self.split = int(self.split_percent*len(self.final_vec))
        self.close_train = self.final_vec[:self.split]
        self.close_test = self.final_vec[self.split:]
        self.inputd=[]
        self.outd=[]
        self.look_back = look_back
        self.num_epoch=num_epoch
        self.losse=losse
        self.met=met
        self.name_coin=name_coin
        self.timep=timep
        self.whatlearn=whatlearn


    def prep_vec(self):
        for ip in range(int(len(self.final_vec)-self.look_back)-1):
            self.inputd.append(self.final_vec[ip:ip+self.look_back])
            self.outd.append(self.final_vec[ip+1:ip+self.look_back+1])

#+-------------------------------------------------------model 2
    def get_model(self):
        val = Input(shape=(self.look_back,1), dtype='float32', name='post')
        yr1 = layers.Conv1D(10, (3),padding='same', activation='relu', input_shape=(self.look_back,1))(val)
        yr = layers.Conv1D(10, (5),padding='same', activation='relu')(yr1)
        cr1 = layers.Conv1D(10, (7), padding='same', activation='relu')(yr) 
        cr = layers.Conv1D(10, (9), padding='same', activation='relu')(cr1)
        crs=layers.add([cr, yr1])
        crs1 = layers.Conv1D(10, (7),padding='same', activation='relu')(crs)
        yr1 = layers.Conv1D(10, (5),padding='same', activation='relu')(yr) 
        yr = layers.add([yr1, yr])                           
        yr2=layers.Conv1D(10, (3),padding='same', activation='relu')(yr)
        yr3 = layers.add([yr2, crs1])
        yr4= layers.Conv1D(1, (3),padding='same', activation='tanh')(yr3)
        yr5=layers.Dense(1)(yr4)
        #yr6=layers.Dense(10)(yr5)
        model = Model(val, yr5)
        return model

    def train_model(self,model):
        model.summary()
        model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=self.losse,metrics=[self.met])
        t1=np.array(self.inputd).reshape(len(self.inputd),self.look_back,1)
        t2=np.array(self.outd).reshape(len(self.inputd),self.look_back,1)
        print("medidas de entrenamiento")
        model.fit(t1,t2,self.look_back,self.num_epoch,verbose=1,validation_split=0.05)
        return model

    def save_model(self,model):
        model_json = model.to_json()
        with open(self.name_coin+self.timep+self.whatlearn+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        #model.save_weights("model.h5")
        model.save_weights(self.name_coin+self.timep+self.whatlearn+".h5")
        print("Saved model to disk")

#+--------------------------------------------------------training
#model.fit(np.array(inputd).reshape(len(inputd),1), np.array(outd), epochs=num_epochs, verbose=1)
    def test_predict(self,model,vect_to_predic):
        predi=[]
        prediction = model.predict(np.array(vect_to_predic).reshape(1,self.look_back,1))
        for ip in range(self.look_back):
            print(np.mean(prediction[0][ip]))
            predi.append(round(np.mean(prediction[0][ip])))
        return predi


    def load_model(self):
        json_file = open(self.name_coin+self.timep+self.whatlearn+".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.name_coin+self.timep+self.whatlearn+".h5")
        print("Loaded model from disk")
        return loaded_model




##################################################################
########################## connection ############################
#+----------------------------------------------------------------
class Ob_fin():
    def __init__(self, coin, timep, user, trader, password,year,month,day, timeframe, nsamp,splt, look_back, epoch, statistic, whatlearn):   ## "NZDJPY", "H1", 64692122,"XMGlobal-MT5 2","Cenidet2015",2021, 8, 6, mt5.TIMEFRAME_H1, 40 , 0.8, 10, 1000
        self.coin=coin
        self.timep=timep
        self.user=user
        self.trader=trader
        self.password=password
        self.year=year
        self.month=month
        self.day=day
        self.timeframe=timeframe
        self.nsamp=nsamp
        self.splt=splt
        self.look_back=look_back
        self.epoch=epoch
        self.statistic=statistic
        self.whatlearn=whatlearn

    def give_datos(self):
        con=Get_connection(self.user,self.trader,self.password)
        con.Get_conn()
        # data
        dta=Get_data(self.year, self.month, self.day, self.coin, self.timeframe, self.nsamp) ### year_month_day
        print(dta.get_rates())
        datos=dta.get_rates()
        con.Mt5_down()
        ####################### para solucionar el pase de elemento se emlea el concepto de herencia y polimorfismo en python 
        if(self.statistic):
            statistic1=Get_statistic(datos,self.coin+"_"+self.timep)  ####### inicializa la clase
            statistic1.statistical_coins(statistic1.rates_toratesframe())

        return datos
###################################################################
###################################################################

    def get_vec(self,datos):
        svec=Vector_to_train(datos, self.coin+'_'+self.timep, 5, self.whatlearn)
        final_vec=svec.get_train_vector(Vector_to_train.get_sum,Vector_to_train.get_h,Vector_to_train.create_arrays)
        print(final_vec)
        return final_vec

##########################################################################################################################################
#statistical_coins(2018, 1, 10, "EURUSD", mt5.TIMEFRAME_H4, 10000)
#statistical_coins(2018, 1, 10, "USDJPY", mt5.TIMEFRAME_H4, 10000)
##########################################################################################################################################
#+-------------------- for training -------------------------------------------------------------- final_vec, istrain = 1 -> train
    def get_data_and_predict(self, final_vec, istrain):
        iapply=Data_learning(self.coin, self.timep, final_vec ,self.splt, self.look_back, self.epoch ,"mean_squared_error", 'accuracy', self.whatlearn)

        if istrain==1:
            iapply.prep_vec()
            modeli=iapply.get_model()
            modeli=iapply.train_model(modeli)
            iapply.save_model(modeli)
        #print(final_vec[len(final_vec)-10:len(final_vec)])
        #print(datos[len(final_vec)-10:len(final_vec)])
        ssf=final_vec[len(final_vec)-self.look_back:len(final_vec)]
        loaded_mod=iapply.load_model()
        predicho=iapply.test_predict(loaded_mod,ssf)  ####### vector del tamano de look_back
        return predicho

    def train_and_save_model(self, final_vec):    #### final_vec ,0.8, 10, 1000
        iapply=Data_learning(self.coin, self.timep, final_vec , self.splt, self.look_back, self.epoch ,"mean_squared_error", 'accuracy', self.whatlearn)
        iapply.prep_vec()
        modeli=iapply.get_model()
        modeli=iapply.train_model(modeli)
        iapply.save_model(modeli)



    def operate(self,arrpr):
        strcp=str(arrpr[len(arrpr)-2])+str(arrpr[len(arrpr)-1])
        print(strcp)
        advice="visual"
        if strcp=="00":
            advice="bajando"
        if strcp=="01":
            advice="comprar"
        if strcp=="11":
            advice="minimo-estacionario"
        if strcp=="10":
            advice="vender"
        if strcp=="12":
            advice="comprar"
        if strcp=="22":
            advice="subiendo"
        if strcp=="23":
            advice="vender"
        if strcp=="30":
            advice="vender"
        if strcp=="32":
            advice="comprar"
        if strcp=="33":
            advice="maximo-estacionario"
        return advice


######################### cuando una clase hereda de otra 
"""
class Perro(Animal):
    def __init__(self, especie, edad, dueño):
        # Alternativa 1
        # self.especie = especie
        # self.edad = edad
        # self.dueño = dueño

        # Alternativa 2
        super().__init__(especie, edad)
        self.dueño = dueño
"""





class Get_train_predict():
    def __init__(self,coin_pairs, istrain, timep, user, trader, password,year,month,day, timeframe, nsamp,splt, look_back, epoch, statistic, whatlearn):
        self.coin_pairs=coin_pairs
        self.istrain=istrain
        self.timep=timep
        self.user=user
        self.trader=trader
        self.password=password
        self.year=year
        self.month=month
        self.day=day
        self.timeframe=timeframe
        self.nsamp=nsamp
        self.splt=splt
        self.look_back=look_back
        self.epoch=epoch
        self.statistic=statistic
        self.whatlearn=whatlearn


    def tpr_or_predic(self):
        predic=[]
        for ip in range(0,len(self.coin_pairs)):
                                                                                           #year, month, day, timeframe, muestras, split, look_back, epoch,#
            #obf=Ob_fin(self.coin_pairs[ip], "H4", 64692122,"XMGlobal-MT5 2","Cenidet2015", 2021, 8, 7, mt5.TIMEFRAME_H4, 10000, 0.8, 10, 1000)
            obf=Ob_fin(self.coin_pairs[ip], self.timep, self.user,self.trader,self.password, self.year, self.month, self.day, self.timeframe,  self.nsamp, self.splt, self.look_back, self.epoch, self.statistic, self.whatlearn)
            dtos=obf.give_datos()
            dtos=obf.give_datos()
            finv=obf.get_vec(dtos)
            
            if(self.istrain):
                obf.train_and_save_model(finv)

            predicho=obf.get_data_and_predict(finv,0)
            print(predicho)
            
            predic.append(self.coin_pairs[ip])
            predic.append(predicho)
            predic.append(obf.operate(predicho))
        return predic



#+--------------------- Estadistica -------------------------------
#statistic=Get_statistic(datos,coin+"_"+timep)
#statistic.statistical_coins(statistic.rates_toratesframe())

########################## en la funcion vector to train definir el tipo de dato que se va a emplear al entrenar
########################## Vector_to_train():


from datetime import datetime
current_time = datetime.now() 
print(str(current_time.year)+ '_'+str(current_time.month)+ '_'+str(current_time.day)+'_'+str(current_time.hour))

#coin_pairs=["EURUSD","USDJPY","GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"] #istrain, tempo, user, broker, password, year, month, day, timeframe, muestras, split, look_back, epoch, estatistic, whatlearn#
coin_pairs=["EURUSD"]
# los tipos de entrenamiento 'cumul'         : acumulados
#                            'val_h'         : salida alta

################# typedata cumul = range_tick(),  val_h = range_high()
file1 = open("C:/Users/MauROG/Documents/Mau/Master/Tetra 4/actividad 8/prediction.txt","a")
file1.write('##############_'+str(current_time.year)+ '_'+str(current_time.month)+ '_'+str(current_time.day)+'_'+str(current_time.hour)+'_'+str(current_time.minute)+'_################ \n')
file1.close()

ob_pred=Get_train_predict(coin_pairs, True, "H1", 68254071,"XMGlobal-MT5 2","TESTCONTRA", current_time.year, current_time.month, current_time.day, mt5.TIMEFRAME_H4, 20000, 0.8, 10, 5, False, 'cumul')
predic=ob_pred.tpr_or_predic()
file1 = open("C:/Users/MauROG/Documents/Mau/Master/Tetra 4/actividad 8/prediction.txt","a")
file1.write('cumul_H4 \n')
file1.write(str(predic)+'\n')
file1.close()
print(predic)


ob_pred=Get_train_predict(coin_pairs, True, "H1", 68254071,"XMGlobal-MT5 2","TESTCONTRA", current_time.year, current_time.month, current_time.day, mt5.TIMEFRAME_H4, 20000, 0.8, 10, 5, False, 'val_h')
predic=ob_pred.tpr_or_predic()
file1 = open("C:/Users/MauROG/Documents/Mau/Master/Tetra 4/actividad 8/prediction.txt","a")
file1.write('val_h_H4 \n')
file1.write(str(predic)+'\n')
file1.close()
print(predic)



ob_pred=Get_train_predict(coin_pairs, True, "H1", 68254071,"XMGlobal-MT5 2","TESTCONTRA", current_time.year, current_time.month, current_time.day, mt5.TIMEFRAME_H1, 20000, 0.8, 10, 5, False, 'cumul')
predic=ob_pred.tpr_or_predic()
file1 = open("C:/Users/MauROG/Documents/Mau/Master/Tetra 4/actividad 8/prediction.txt","a")
file1.write('cumul_H1 \n')
file1.write(str(predic)+'\n')
file1.close()
print(predic)


ob_pred=Get_train_predict(coin_pairs, True, "H1", 68254071,"XMGlobal-MT5 2","TESTCONTRA", current_time.year, current_time.month, current_time.day, mt5.TIMEFRAME_H1, 20000, 0.8, 10, 5, False, 'val_h')
predic=ob_pred.tpr_or_predic()
file1 = open("C:/Users/MauROG/Documents/Mau/Master/Tetra 4/actividad 8/prediction.txt","a")
file1.write('val_h_H1 \n')
file1.write(str(predic)+'\n')
file1.close()
print(predic)