from flask import Flask, render_template
#from app_model import Pokemon, PokemonTwo, PokemonBattleEnv, PokemonBattleEnvTwo, Team, TeamTwo, DQNAgent, DQNAgentTwo
from keras.models import load_model
import numpy as np
import keras
from keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import adam_v2
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import json
import math
import threading
import random
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import numpy as np
from keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import adam_v2
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import json
import math
import random
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

app = Flask(__name__)

sess = tf.compat.v1.Session()
sess2 = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
graph2 = tf.compat.v1.get_default_graph()


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'PokeBattle'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 540

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False

buff_stage = {"-4":0.5,
               "-3":0.5715,
               "-2":0.6667,
               "-1":0.8,
               "0":1,
               "1":1.25,
               "2":1.5,
               "3":1.75,
               "4":2
}

STAB = 1.2

SuperEffective = 1.6
Not_Very_Effective = 0.625
Immune = 0.390625

time_slot = 270*2

def Damage(atkpk_power,atkpk_ATK,defpk_DEF):
    return math.floor(0.5*atkpk_power*atkpk_ATK*1.3/defpk_DEF)+1

def Switch_poke_decide(battle_able_list, opponent):
    with open('type.json', 'r', encoding="utf-8") as ty:
        ttype = json.load(ty)
        
    prefer_idx = 0
    prefer_metric = 0
    max_prefer_metric = 0
    for i in range(0, len(battle_able_list)):
        for k in range(0,2):
            for j in range(0,2):
                if ttype[f'{battle_able_list[i].type[k]}'][f'{opponent.type[j]}'] == 1.6:
                    prefer_metric += 1
                if ttype[f'{battle_able_list[i].type[k]}'][f'{opponent.type[j]}'] == 0.625:
                    prefer_metric -= 1
                if ttype[f'{battle_able_list[i].type[k]}'][f'{opponent.type[j]}'] == 0.391:
                    prefer_metric -= 1.5

        for j in range(0,2):
                if ttype[f'{battle_able_list[i].pk_fm_type}'][f'{opponent.type[j]}'] == 1.6:
                    prefer_metric += 1
                if ttype[f'{battle_able_list[i].pk_fm_type}'][f'{opponent.type[j]}'] == 0.625:
                    prefer_metric -= 1
                if ttype[f'{battle_able_list[i].pk_fm_type}'][f'{opponent.type[j]}'] == 0.391:
                    prefer_metric -= 1.5

                if ttype[f'{battle_able_list[i].pk_cm1_type}'][f'{opponent.type[j]}'] == 1.6:
                    prefer_metric += 1
                if ttype[f'{battle_able_list[i].pk_cm1_type}'][f'{opponent.type[j]}'] == 0.625:
                    prefer_metric -= 1
                if ttype[f'{battle_able_list[i].pk_cm1_type}'][f'{opponent.type[j]}'] == 0.391:
                    prefer_metric -= 1.5

                if ttype[f'{battle_able_list[i].pk_cm2_type}'][f'{opponent.type[j]}'] == 1.6:
                    prefer_metric += 1
                if ttype[f'{battle_able_list[i].pk_cm2_type}'][f'{opponent.type[j]}'] == 0.625:
                    prefer_metric -= 1
                if ttype[f'{battle_able_list[i].pk_cm2_type}'][f'{opponent.type[j]}'] == 0.391:
                    prefer_metric -= 1.5
        
        if prefer_metric > max_prefer_metric:
            max_prefer_metric = prefer_metric
            prefer_idx = i
                
    return prefer_idx

class Team:
    def __init__(self, pkList, teamID, CP_limit):
        self.countT = time_slot
        self.firstOrder = Pokemon(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.secondOrder = Pokemon(pkList[1][0], pkList[1][1], pkList[1][2], pkList[1][3], CP_limit)
        self.thirdOrder = Pokemon(pkList[2][0], pkList[2][1], pkList[2][2], pkList[2][3], CP_limit)
        self.TeamID = teamID
        self.firstOrder.OnStage = True
        self.battling = Pokemon(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.battling.last_action_damage = 0
        self.shield = 2
        self.switchClock = 60*2
        self.receivedCM = False
        self.CP_limit = CP_limit
        self.fault = False


    def reset(self, pkList, teamID, CP_limit):
        self.countT = time_slot
        self.firstOrder = Pokemon(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.secondOrder = Pokemon(pkList[1][0], pkList[1][1], pkList[1][2], pkList[1][3], CP_limit)
        self.thirdOrder = Pokemon(pkList[2][0], pkList[2][1], pkList[2][2], pkList[2][3], CP_limit)
        self.TeamID = teamID
        self.firstOrder.OnStage = True
        self.battling = Pokemon(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.battling.last_action_damage = 0
        self.shield = 2
        self.switchClock = 60*2
        self.receivedCM = False
        self.CP_limit = CP_limit
        self.fault = False

    def fast_move(self, other):
        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)

        with open('moves.json', 'r', encoding="utf-8") as mv:
            mov = json.load(mv)
        for move_data in mov:
            if move_data.get('moveId') == self.battling.fm:
                self.battling.pk_fm_power =  move_data.get('power')
        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_fm_power *= SuperEffective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_fm_power *= Not_Very_Effective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_fm_power *= Immune
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 1:
            self.battling.pk_fm_power *= 1

        if ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_fm_power *= SuperEffective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_fm_power *= Not_Very_Effective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_fm_power *= Immune
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 1:
            self.battling.pk_fm_power *= 1
        
        if self.battling.pk_fm_type == self.battling.pk_type1:
            self.battling.pk_fm_power *= STAB
        if self.battling.pk_fm_type == self.battling.pk_type2:
            self.battling.pk_fm_power *= STAB

        self.countT -= self.battling.pk_fm_turn
        return self.battling.pk_fm_energyGain, Damage(self.battling.pk_fm_power,self.battling.Atk, other.battling.Def)
    
    def charged_move1(self, other):
        other.receivedCM = True

        atk_buff = 0
        def_buff = 0
        buff_target = None

        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)
        with open('moves.json', 'r', encoding="utf-8") as mv:
            mov = json.load(mv)
        for move_data in mov:
            if move_data.get('moveId') == self.battling.cm1:
                self.battling.pk_cm1_power =  move_data.get('power')
                if "buffs" in move_data:
                    atk_buff = move_data.get('buffs')[0]
                    def_buff = move_data.get('buffs')[1]
                    buff_target = move_data.get('buffTarget')
        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_cm1_power *= SuperEffective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_cm1_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_cm1_power *= Immune
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 1:
            self.battling.pk_cm1_power *= 1

        if ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_cm1_power *= SuperEffective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_cm1_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_cm1_power *= Immune
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 1:
            self.battling.pk_cm1_power *= 1

        if self.battling.pk_cm1_type == self.battling.pk_type1:
            self.battling.pk_cm1_power *= STAB
        if self.battling.pk_cm1_type == self.battling.pk_type2:
            self.battling.pk_cm1_power *= STAB
        self.countT -= 20
        return self.battling.pk_cm1_energyLoss, Damage(self.battling.pk_cm1_power,self.battling.Atk, other.battling.Def), atk_buff, def_buff, buff_target

    def charged_move2(self, other):
        other.receivedCM = True

        atk_buff = 0
        def_buff = 0
        buff_target = None

        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)
        with open('moves.json', 'r', encoding="utf-8") as mv:
            mov = json.load(mv)
        for move_data in mov:
            if move_data.get('moveId') == self.battling.cm2:
                self.battling.pk_cm2_power =  move_data.get('power')
                if "buffs" in move_data:
                    atk_buff = move_data.get('buffs')[0]
                    def_buff = move_data.get('buffs')[1]
                    buff_target = move_data.get('buffTarget')
        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_cm2_power *= SuperEffective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_cm2_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_cm2_power *= Immune
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 1:
            self.battling.pk_cm2_power *= 1

        if ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_cm2_power *= SuperEffective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_cm2_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_cm2_power *= Immune
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 1:
            self.battling.pk_cm2_power *= 1

        if self.battling.pk_cm2_type == self.battling.pk_type1:
            self.battling.pk_cm2_power *= STAB
        if self.battling.pk_cm2_type == self.battling.pk_type2:
            self.battling.pk_cm2_power *= STAB
        self.countT -= 20
        return self.battling.pk_cm2_energyLoss, Damage(self.battling.pk_cm2_power,self.battling.Atk, other.battling.Def), atk_buff, def_buff, buff_target 

class TeamTwo:
    def __init__(self, pkList, teamID, CP_limit):
        self.countT = time_slot
        self.firstOrder = PokemonTwo(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.secondOrder = PokemonTwo(pkList[1][0], pkList[1][1], pkList[1][2], pkList[1][3], CP_limit)
        self.thirdOrder = PokemonTwo(pkList[2][0], pkList[2][1], pkList[2][2], pkList[2][3], CP_limit)
        self.TeamID = teamID
        self.firstOrder.OnStage = True
        self.battling = PokemonTwo(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.battling.last_action_damage = 0
        self.shield = 2
        self.switchClock = 60*2
        self.receivedCM = False
        self.CP_limit = CP_limit
        self.fault = False


    def reset(self, pkList, teamID, CP_limit):
        self.countT = time_slot
        self.firstOrder = PokemonTwo(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.secondOrder = PokemonTwo(pkList[1][0], pkList[1][1], pkList[1][2], pkList[1][3], CP_limit)
        self.thirdOrder = PokemonTwo(pkList[2][0], pkList[2][1], pkList[2][2], pkList[2][3], CP_limit)
        self.TeamID = teamID
        self.firstOrder.OnStage = True
        self.battling = PokemonTwo(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.battling.last_action_damage = 0
        self.shield = 2
        self.switchClock = 60*2
        self.receivedCM = False
        self.CP_limit = CP_limit
        self.fault = False

        

    def action(self, other, choice):
        '''
        Gives us 4 total movement options.
        0: 小招
        1: 大招一
        2: 大招二
        3: 停頓不動作
        4: 換人
        5: 開盾
        '''
        '''elif choice == 5:
            self.shield -= 1
            self.battling.HP += other.battling.last_action_damage-1'''
            

    def fast_move(self, other):
        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)
        with open('moves.json', 'r', encoding="utf-8") as mv:
            mov = json.load(mv)
        for move_data in mov:
            if move_data.get('moveId') == self.battling.fm:
                self.battling.pk_fm_power =  move_data.get('power')
        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_fm_power *= SuperEffective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_fm_power *= Not_Very_Effective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_fm_power *= Immune
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 1:
            self.battling.pk_fm_power *= 1

        if ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_fm_power *= SuperEffective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_fm_power *= Not_Very_Effective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_fm_power *= Immune
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 1:
            self.battling.pk_fm_power *= 1
        
        if self.battling.pk_fm_type == self.battling.pk_type1:
            self.battling.pk_fm_power *= STAB
        if self.battling.pk_fm_type == self.battling.pk_type2:
            self.battling.pk_fm_power *= STAB

        self.countT -= self.battling.pk_fm_turn
        return self.battling.pk_fm_energyGain, Damage(self.battling.pk_fm_power,self.battling.Atk, other.battling.Def)
    
    def charged_move1(self, other):
        other.receivedCM = True

        atk_buff = 0
        def_buff = 0
        buff_target = None

        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)
        with open('moves.json', 'r', encoding="utf-8") as mv:
            mov = json.load(mv)
        for move_data in mov:
            if move_data.get('moveId') == self.battling.cm1:
                self.battling.pk_cm1_power =  move_data.get('power')
                if "buffs" in move_data:
                    atk_buff = move_data.get('buffs')[0]
                    def_buff = move_data.get('buffs')[1]
                    buff_target = move_data.get('buffTarget')

        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_cm1_power *= SuperEffective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_cm1_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_cm1_power *= Immune
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 1:
            self.battling.pk_cm1_power *= 1

        if ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_cm1_power *= SuperEffective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_cm1_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_cm1_power *= Immune
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 1:
            self.battling.pk_cm1_power *= 1

        if self.battling.pk_cm1_type == self.battling.pk_type1:
            self.battling.pk_cm1_power *= STAB
        if self.battling.pk_cm1_type == self.battling.pk_type2:
            self.battling.pk_cm1_power *= STAB

        self.countT -= 20
        return self.battling.pk_cm1_energyLoss, Damage(self.battling.pk_cm1_power,self.battling.Atk, other.battling.Def), atk_buff, def_buff, buff_target

    def charged_move2(self, other):
        other.receivedCM = True

        atk_buff = 0
        def_buff = 0
        buff_target = None

        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)
        with open('moves.json', 'r', encoding="utf-8") as mv:
            mov = json.load(mv)
        for move_data in mov:
            if move_data.get('moveId') == self.battling.cm2:
                self.battling.pk_cm2_power =  move_data.get('power')
                if "buffs" in move_data:
                    atk_buff = move_data.get('buffs')[0]
                    def_buff = move_data.get('buffs')[1]
                    buff_target = move_data.get('buffTarget')

        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_cm2_power *= SuperEffective 
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_cm2_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_cm2_power *= Immune
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 1:
            self.battling.pk_cm2_power *= 1

        if ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_cm2_power *= SuperEffective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_cm2_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_cm2_power *= Immune
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 1:
            self.battling.pk_cm2_power *= 1

        if self.battling.pk_cm2_type == self.battling.pk_type1:
            self.battling.pk_cm2_power *= STAB
        if self.battling.pk_cm2_type == self.battling.pk_type2:
            self.battling.pk_cm2_power *= STAB
        
        self.countT -= 20
        return self.battling.pk_cm2_energyLoss, Damage(self.battling.pk_cm2_power,self.battling.Atk, other.battling.Def), atk_buff, def_buff, buff_target 

class Pokemon:
    def __init__(self, name, pk_fm, pk_cm1, pk_cm2, CP_limit):
        name = name.lower()
        if "暗影" in name:
            self.shadow = True
        elif "shadow" in name:
            self.shadow = True
        else:
            self.shadow = False
        name = name.replace('暗影', '')
        name = name.replace('shadow', '')
        name = name.replace('(', '')
        name = name.replace(')', '')

        self.name = name

        self.TeamID = 0
        self.OnStage = False
        self.OnStageAble = True
        self.fm = pk_fm
        self.cm1 = pk_cm1
        self.cm2 = pk_cm2
        self.last_action_damage = 0
        self.atkBuffLv = 0
        self.defBuffLv = 0

        with open('Pokemon_'+str(CP_limit)+'_default.json', 'r', encoding="utf-8") as json_file:
            all_data = json.load(json_file) 

        self.Atk = all_data[name]['CurrentAtk']
        self.Def = all_data[name]['CurrentDef']
        self.HP= all_data[name]['CurrentHP']
        self.pk_energy = 0

        if self.shadow == True:
            self.Atk *= 6/5
            self.Def *= 5/6

        self.AtkIV = all_data[name]['DefaultAtkIV']
        self.DefIV = all_data[name]['DefaulDefIV']
        self.StaIV = all_data[name]['DefaulStaIV']
        self.Lv = all_data[name]['DefaulLV']

        with open('moves.json', 'r') as m:
            mov = json.load(m)
        for move_data in mov:
            if move_data.get('moveId') == pk_fm:
                self.pk_fm_type =  move_data.get('type')
                self.pk_fm_power =  move_data.get('power')
                self.pk_fm_energyGain =  move_data.get('energyGain')
                self.pk_fm_turn =  move_data.get('cooldown')/500
            elif move_data.get('moveId') == pk_cm1:
                self.pk_cm1_type =  move_data.get('type')
                self.pk_cm1_power =  move_data.get('power')
                self.pk_cm1_energyLoss =  -1*move_data.get('energy')
                self.pk_cm1_turn =  1
            elif move_data.get('moveId') == pk_cm2:
                self.pk_cm2_type =  move_data.get('type')
                self.pk_cm2_power =  move_data.get('power')
                self.pk_cm2_energyLoss =  -1*move_data.get('energy')
                self.pk_cm2_turn =  1

        [self.pk_type1, self.pk_type2]= all_data[name]['type']
        self.type = [self.pk_type1, self.pk_type2]

class PokemonTwo:
    def __init__(self, name, pk_fm, pk_cm1, pk_cm2, CP_limit):
        name = name.lower()
        if "暗影" in name:
            self.shadow = True
        elif "shadow" in name:
            self.shadow = True
        else:
            self.shadow = False
        name = name.replace('暗影', '')
        name = name.replace('shadow', '')
        name = name.replace('(', '')
        name = name.replace(')', '')

        self.name = name

        self.TeamID = 0
        self.OnStage = False
        self.OnStageAble = True
        self.fm = pk_fm
        self.cm1 = pk_cm1
        self.cm2 = pk_cm2
        self.last_action_damage = 0
        self.atkBuffLv = 0
        self.defBuffLv = 0
        

        with open('Pokemon_'+str(CP_limit)+'_default.json', 'r', encoding="utf-8") as json_file:
            all_data = json.load(json_file) 

        self.Atk = all_data[name]['CurrentAtk']
        self.Def = all_data[name]['CurrentDef']
        self.HP= all_data[name]['CurrentHP']
        self.pk_energy = 0
        
        if self.shadow == True:
            self.Atk *= 6/5
            self.Def *= 5/6

        self.AtkIV = all_data[name]['DefaultAtkIV']
        self.DefIV = all_data[name]['DefaulDefIV']
        self.StaIV = all_data[name]['DefaulStaIV']
        self.Lv = all_data[name]['DefaulLV']

        with open('moves.json', 'r') as m:
            mov = json.load(m)
        for move_data in mov:
            if move_data.get('moveId') == pk_fm:
                self.pk_fm_type =  move_data.get('type')
                self.pk_fm_power =  move_data.get('power')
                self.pk_fm_energyGain =  move_data.get('energyGain')
                self.pk_fm_turn =  move_data.get('cooldown')/500
            elif move_data.get('moveId') == pk_cm1:
                self.pk_cm1_type =  move_data.get('type')
                self.pk_cm1_power =  move_data.get('power')
                self.pk_cm1_energyLoss =  -1*move_data.get('energy')
                self.pk_cm1_turn =  1
            elif move_data.get('moveId') == pk_cm2:
                self.pk_cm2_type =  move_data.get('type')
                self.pk_cm2_power =  move_data.get('power')
                self.pk_cm2_energyLoss =  -1*move_data.get('energy')
                self.pk_cm2_turn =  1

        [self.pk_type1, self.pk_type2]= all_data[name]['type']
        self.type = [self.pk_type1, self.pk_type2]

class PokemonBattleEnv:

    SIZE = 10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    OBSERVATION_SPACE_VALUES = np.expand_dims(OBSERVATION_SPACE_VALUES, axis = -1)
    ACTION_SPACE_SIZE = 6
    

    def reset(self, team):
        self.team = team
        self.team_switch_clock = 60*2
        self.team_shield = 2
        self.shieldusing = True
        self.switching = True
        self.DD = 0
        self.EE = 0
        self.WAIT = 0
        self.team.fault = False
        self.move_buff_target = None
        self.move_atk_buff = 0
        self.move_def_buff = 0
        return 1

    #Team的這一步做了甚麼、做得好不好
    def step(self, other, action):
        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)

        PK_opp = 1
        PK_slf = 1
        PK_slf_oppfm = 1
        PK_slf_oppcm1 = 1
        PK_slf_oppcm2 = 1

        PK_opp_slffm = 1
        PK_opp_slfcm1 = 1
        PK_opp_slfcm2 = 1

        reward = 1
        done = False
        
        #不確定other, self的用法
        pokemon_list = [self.team.firstOrder, self.team.secondOrder, self.team.thirdOrder] 
        #換人冷卻計時+1
        if self.team.battling.HP <= 0:
            self.team.receivedCM = False
            #print("需要換人")
            #搜尋可上場的替換角色
            battle_able = []
            battle_able_list = []
            if self.team.firstOrder.OnStageAble == True and self.team.firstOrder.OnStage == False and self.team.firstOrder !=self.team.battling:
                battle_able.append(0)
                battle_able_list.append(self.team.firstOrder)
            if self.team.secondOrder.OnStageAble == True and self.team.secondOrder.OnStage == False and self.team.secondOrder !=self.team.battling:
                battle_able.append(1)
                battle_able_list.append(self.team.secondOrder)
            if self.team.thirdOrder.OnStageAble == True and self.team.thirdOrder.OnStage == False and self.team.thirdOrder !=self.team.battling:
                battle_able.append(2)
                battle_able_list.append(self.team.thirdOrder)


            #先隨機挑角色上場，機制後續須補上
            #如果沒有可使用角色，則隊伍fault
            if not battle_able:
                self.team.fault = True
                #print("Team "+ f"{self.team.TeamID}"+" 全部角色皆陣亡！")
            else:
                s = battle_able[Switch_poke_decide(battle_able_list, other.team.battling)]
                if self.team.firstOrder.name == self.team.battling.name:
                    self.team.firstOrder.OnStage = False
                    self.team.firstOrder.HP = self.team.battling.HP
                    self.team.firstOrder.pk_energy = self.team.battling.pk_energy
                    self.team.firstOrder.OnStageAble = False
                    self.team.firstOrder.move_buff_target = None
                    self.team.firstOrder.move_atk_buff = 0
                    self.team.firstOrder.move_def_buff = 0
                elif self.team.secondOrder.name == self.team.battling.name:
                    self.team.secondOrder.OnStage = False
                    self.team.secondOrder.HP = self.team.battling.HP
                    self.team.secondOrder.pk_energy = self.team.battling.pk_energy
                    self.team.secondOrder.OnStageAble = False
                    self.team.secondOrder.move_buff_target = None
                    self.team.secondOrder.move_atk_buff = 0
                    self.team.secondOrder.move_def_buff = 0
                elif self.team.thirdOrder.name == self.team.battling.name:
                    self.team.thirdOrder.OnStage = False
                    self.team.thirdOrder.HP = self.team.battling.HP
                    self.team.thirdOrder.pk_energy = self.team.battling.pk_energy
                    self.team.thirdOrder.OnStageAble = False
                    self.team.thirdOrder.move_buff_target = None
                    self.team.thirdOrder.move_atk_buff = 0
                    self.team.thirdOrder.move_def_buff = 0

                if s == 0:
                    self.team.battling = self.team.firstOrder
                    self.team.firstOrder.OnStage = True
                elif s == 1:
                    self.team.battling = self.team.secondOrder
                    self.team.secondOrder.OnStage = True
                elif s == 2:
                    self.team.battling = self.team.thirdOrder
                    self.team.thirdOrder.OnStage = True

            self.DD = 0
            self.EE = 0
            self.WAIT = random.randint(6, 20)
            other.DD = 0
            other.EE = 0
            if other.team.battling.HP <= 0:
                M = max(self.WAIT, other.WAIT)
                self.WAIT = M
                other.WAIT = M
                
            self.team.countT -= self.WAIT
            other.team.countT = self.team.countT
            
            #print("Team "+ f"{self.team.TeamID}"+'換人等待 '+f'{self.WAIT}'+' 秒')
            
        elif action == 4 and self.team.countT-1 == time_slot:
            other.DD = 0
            self.EE = 0

            battle_able = []
            battle_able_list = []
            if self.team.firstOrder.OnStageAble == True and self.team.firstOrder.OnStage == False and self.team.firstOrder !=self.team.battling:
                battle_able.append(0)
                battle_able_list.append(self.team.firstOrder)
            if self.team.secondOrder.OnStageAble == True and self.team.secondOrder.OnStage == False and self.team.secondOrder !=self.team.battling:
                battle_able.append(1)
                battle_able_list.append(self.team.secondOrder)
            if self.team.thirdOrder.OnStageAble == True and self.team.thirdOrder.OnStage == False and self.team.thirdOrder !=self.team.battling:
                battle_able.append(2)
                battle_able_list.append(self.team.thirdOrder)


            #先隨機挑角色上場，機制後續須補上
            #如果沒有可使用角色，則隊伍fault
            if not battle_able:
                self.team.fault = True
                #print("Team "+ f"{self.team.TeamID}"+" 全部角色皆陣亡！")
            else:
                s = battle_able[Switch_poke_decide(battle_able_list, other.team.battling)]
            
                if self.team.firstOrder.name == self.team.battling.name:
                    self.team.firstOrder.OnStage = False
                    self.team.firstOrder.HP = self.team.battling.HP
                    self.team.firstOrder.pk_energy = self.team.battling.pk_energy
                    self.team.firstOrder.OnStageAble = True
                    self.team.firstOrder.move_buff_target = None
                    self.team.firstOrder.move_atk_buff = 0
                    self.team.firstOrder.move_def_buff = 0
                if self.team.secondOrder.name == self.team.battling.name:
                    self.team.secondOrder.OnStage = False
                    self.team.secondOrder.HP = self.team.battling.HP
                    self.team.secondOrder.pk_energy = self.team.battling.pk_energy
                    self.team.secondOrder.OnStageAble = True
                    self.team.secondOrder.move_buff_target = None
                    self.team.secondOrder.move_atk_buff = 0
                    self.team.secondOrder.move_def_buff = 0
                if self.team.thirdOrder.name == self.team.battling.name:
                    self.team.thirdOrder.OnStage = False
                    self.team.thirdOrder.HP = self.team.battling.HP
                    self.team.thirdOrder.pk_energy = self.team.battling.pk_energy
                    self.team.thirdOrder.OnStageAble = True
                    self.team.thirdOrder.move_buff_target = None
                    self.team.thirdOrder.move_atk_buff = 0
                    self.team.thirdOrder.move_def_buff = 0

                if s == 0:
                    self.team.battling = self.team.firstOrder
                    self.team.firstOrder.OnStage = True
                if s == 1:
                    self.team.battling = self.team.secondOrder
                    self.team.secondOrder.OnStage = True
                if s == 2:
                    self.team.battling = self.team.thirdOrder
                    self.team.thirdOrder.OnStage = True
            self.WAIT = 0
            self.team.countT -= 1
            self.team.switchClock = 0 
            #print('換上 '+f'{self.team.battling}')
        elif self.team.countT-1 == time_slot:
            if action == 0:
                self.WAIT = 0
                self.EE, other.DD = self.team.fast_move(other.team)
            elif action == 1:
                self.WAIT = 0
                self.EE, other.DD, self.move_atk_buff, self.move_def_buff, self.move_buff_target = self.team.charged_move1(other.team)
                other.team.receivedCM = True
                other.team.countT = self.team.countT
                #print("Team "+ f"{self.team.TeamID}"+" 使用大招一, 造成傷害："+f'{other.DD}')
            elif action == 2:
                self.WAIT = 0
                self.EE, other.DD, self.move_atk_buff, self.move_def_buff, self.move_buff_target = self.team.charged_move2(other.team)
                other.team.receivedCM = True
                other.team.countT = self.team.countT
                #print("Team "+ f"{self.team.TeamID}"+" 使用大招二, 造成傷害："+f"{other.DD}")
 
        if self.team.receivedCM == True and self.team.shield >= 1 and self.team.countT == time_slot:
            if action == 5:
                other.DD = 1
                self.team.shield -= 1
                
                #print("Team "+ f"{self.team.TeamID}"+" 開盾擋下了攻擊！")
            self.team.receivedCM = False
        elif self.team.receivedCM == True and self.team.shield ==0 and self.team.countT == time_slot:
            self.team.receivedCM = False

        if time_slot == self.team.countT:
            self.team.battling.pk_energy += self.EE
            if self.team.battling.pk_energy >= 100:
                self.team.battling.pk_energy = 100
            other.team.battling.HP -= other.DD
            

            self.team.switchClock += self.WAIT
            other.team.switchClock += self.WAIT
            self.team.receivedCM = False
            other.DD = 0
            self.EE = 0
            self.WAIT = 0

            if self.move_buff_target == 'self':
                #reset atk and def to non-buff state
                if self.team.battling.atkBuffLv != 0:
                    self.team.battling.Atk /= buff_stage.get(str(self.team.battling.atkBuffLv))
                if self.team.battling.defBuffLv != 0:
                    self.team.battling.Def /= buff_stage.get(str(self.team.battling.defBuffLv))
                #new buff state
                self.team.battling.atkBuffLv += self.move_atk_buff
                self.team.battling.defBuffLv += self.move_def_buff
                if self.team.battling.atkBuffLv > 4: self.team.battling.atkBuffLv = 4
                if self.team.battling.atkBuffLv < -4: self.team.battling.atkBuffLv = -4
                if self.team.battling.defBuffLv > 4: self.team.battling.defBuffLv = 4
                if self.team.battling.defBuffLv < -4: self.team.battling.defBuffLv = -4
                #update atk and def
                self.team.battling.Atk *= buff_stage.get(str(self.team.battling.atkBuffLv))
                self.team.battling.Def *= buff_stage.get(str(self.team.battling.defBuffLv))

            if self.move_buff_target == 'opponent':
                #reset atk and def to non-buff state
                if other.team.battling.atkBuffLv != 0:
                    other.team.battling.Atk /= buff_stage.get(str(other.team.battling.atkBuffLv))
                if other.team.battling.defBuffLv != 0:
                    other.team.battling.Def /= buff_stage.get(str(other.team.battling.defBuffLv))
                #new buff state
                other.team.battling.atkBuffLv += self.move_atk_buff
                other.team.battling.defBuffLv += self.move_def_buff
                if other.team.battling.atkBuffLv > 4: other.team.battling.atkBuffLv = 4
                if other.team.battling.atkBuffLv < -4: other.team.battling.atkBuffLv = -4
                if other.team.battling.defBuffLv > 4: other.team.battling.defBuffLv = 4
                if other.team.battling.defBuffLv < -4: other.team.battling.defBuffLv = -4
                #update atk and def
                other.team.battling.Atk *= buff_stage.get(str(other.team.battling.atkBuffLv))
                other.team.battling.Def *= buff_stage.get(str(other.team.battling.defBuffLv))

            self.move_buff_target = None
            self.move_atk_buff = 0
            self.move_def_buff = 0


        

        # 己方屬性是否劣勢，是則PK_TYPE_REWARD>0
        if self.team.fault != True and other.team.fault != True:
            for i in range(0,2):
                for j in range(0,2):
                    PK_slf *=  ttype[f'{other.team.battling.type[i]}'][f'{self.team.battling.type[j]}']
                    PK_slf_oppfm *=  ttype[f'{other.team.battling.pk_fm_type}'][f'{self.team.battling.type[j]}']
                    PK_slf_oppcm1 *=  ttype[f'{other.team.battling.pk_cm1_type}'][f'{self.team.battling.type[j]}']
                    PK_slf_oppcm2 *=  ttype[f'{other.team.battling.pk_cm2_type}'][f'{self.team.battling.type[j]}']
                    

            # 敵方屬性是否劣勢，是則PK_TYPE_REWARD>0
            for i in range(0,2):
                for j in range(0,2):
                    PK_opp *=  ttype[f'{self.team.battling.type[i]}'][f'{other.team.battling.type[j]}']
                    PK_opp_slffm *=  ttype[f'{self.team.battling.pk_fm_type}'][f'{other.team.battling.type[j]}']
                    PK_opp_slfcm1 *=  ttype[f'{self.team.battling.pk_cm1_type}'][f'{other.team.battling.type[j]}']
                    PK_opp_slfcm2 *=  ttype[f'{self.team.battling.pk_cm2_type}'][f'{other.team.battling.type[j]}']
            
            next_state = [PK_opp, PK_slf]
            if PK_opp/PK_slf >= 1:
                reward *= PK_opp/PK_slf/3
            else:
                reward *= -1*PK_slf/PK_opp/3
            reward -= (PK_slf_oppfm + PK_slf_oppcm1 + PK_slf_oppcm2)
            reward += (PK_opp_slffm + PK_opp_slfcm1 + PK_opp_slfcm2)

            with open('Pokemon_'+f'{self.team.CP_limit}'+'_default.json', 'r', encoding="utf-8") as json_file:
                all_data = json.load(json_file)

            reward += (self.team.battling.HP/all_data[f'{self.team.battling.name}']['CurrentHP'] - other.team.battling.HP/all_data[f'{other.team.battling.name}']['CurrentHP'])*2
            
            reward += (self.team.battling.pk_energy/100 - other.team.battling.pk_energy/100)*2
            if other.team.battling.atkBuffLv < 0: reward -= 1
            if other.team.battling.defBuffLv < 0: reward -= 1
            if self.team.battling.atkBuffLv > 0: reward += 1
            if self.team.battling.defBuffLv > 0: reward += 1
            next_state = reward
            return next_state, reward, done, action
        else: 
            return reward, reward, done, action

class PokemonBattleEnvTwo:

    SIZE = 10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    OBSERVATION_SPACE_VALUES = np.expand_dims(OBSERVATION_SPACE_VALUES, axis = -1)
    ACTION_SPACE_SIZE = 6
    

    def reset(self, team):
        self.team = team
        self.team_switch_clock = 60*2
        self.team_shield = 2
        self.shieldusing = True
        self.switching = True
        self.DD = 0
        self.EE = 0
        self.WAIT = 0
        self.team.fault = False
        self.move_buff_target = None
        self.move_atk_buff = 0
        self.move_def_buff = 0
        return 1

    #Team的這一步做了甚麼、做得好不好
    def step(self, other, action):
        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)

        PK_opp = 1
        PK_slf = 1
        PK_slf_oppfm = 1
        PK_slf_oppcm1 = 1
        PK_slf_oppcm2 = 1

        PK_opp_slffm = 1
        PK_opp_slfcm1 = 1
        PK_opp_slfcm2 = 1

        reward = 1
        done = False
        
        #不確定other, self的用法
        pokemon_list = [self.team.firstOrder, self.team.secondOrder, self.team.thirdOrder] 
        #換人冷卻計時+1
        if self.team.battling.HP <= 0:
            self.team.receivedCM = False
            #print("需要換人")
            #搜尋可上場的替換角色
            battle_able = []
            battle_able_list = []
            if self.team.firstOrder.OnStageAble == True and self.team.firstOrder.OnStage == False and self.team.firstOrder !=self.team.battling:
                battle_able.append(0)
                battle_able_list.append(self.team.firstOrder)
            if self.team.secondOrder.OnStageAble == True and self.team.secondOrder.OnStage == False and self.team.secondOrder !=self.team.battling:
                battle_able.append(1)
                battle_able_list.append(self.team.secondOrder)
            if self.team.thirdOrder.OnStageAble == True and self.team.thirdOrder.OnStage == False and self.team.thirdOrder !=self.team.battling:
                battle_able.append(2)
                battle_able_list.append(self.team.thirdOrder)


            #先隨機挑角色上場，機制後續須補上
            #如果沒有可使用角色，則隊伍fault
            if not battle_able:
                self.team.fault = True
                #print("Team "+ f"{self.team.TeamID}"+" 全部角色皆陣亡！")
            else:
                s = battle_able[Switch_poke_decide(battle_able_list, other.team.battling)]
                if self.team.firstOrder.name == self.team.battling.name:
                    self.team.firstOrder.OnStage = False
                    self.team.firstOrder.HP = self.team.battling.HP
                    self.team.firstOrder.pk_energy = self.team.battling.pk_energy
                    self.team.firstOrder.OnStageAble = False
                    self.team.firstOrder.move_buff_target = None
                    self.team.firstOrder.move_atk_buff = 0
                    self.team.firstOrder.move_def_buff = 0
                elif self.team.secondOrder.name == self.team.battling.name:
                    self.team.secondOrder.OnStage = False
                    self.team.secondOrder.HP = self.team.battling.HP
                    self.team.secondOrder.pk_energy = self.team.battling.pk_energy
                    self.team.secondOrder.OnStageAble = False
                    self.team.secondOrder.move_buff_target = None
                    self.team.secondOrder.move_atk_buff = 0
                    self.team.secondOrder.move_def_buff = 0
                elif self.team.thirdOrder.name == self.team.battling.name:
                    self.team.thirdOrder.OnStage = False
                    self.team.thirdOrder.HP = self.team.battling.HP
                    self.team.thirdOrder.pk_energy = self.team.battling.pk_energy
                    self.team.thirdOrder.OnStageAble = False
                    self.team.thirdOrder.move_buff_target = None
                    self.team.thirdOrder.move_atk_buff = 0
                    self.team.thirdOrder.move_def_buff = 0

                if s == 0:
                    self.team.battling = self.team.firstOrder
                    self.team.firstOrder.OnStage = True
                elif s == 1:
                    self.team.battling = self.team.secondOrder
                    self.team.secondOrder.OnStage = True
                elif s == 2:
                    self.team.battling = self.team.thirdOrder
                    self.team.thirdOrder.OnStage = True

            self.DD = 0
            self.EE = 0
            self.WAIT = random.randint(6, 20)
            other.DD = 0
            other.EE = 0
            if other.team.battling.HP <= 0:
                M = max(self.WAIT, other.WAIT)
                self.WAIT = M
                other.WAIT = M
                
            self.team.countT -= self.WAIT
            other.team.countT = self.team.countT
            
            #print("Team "+ f"{self.team.TeamID}"+'換人等待 '+f'{self.WAIT}'+' 秒')
            
        elif action == 4 and self.team.countT-1 == time_slot:
            other.DD = 0
            self.EE = 0

            battle_able = []
            battle_able_list = []
            if self.team.firstOrder.OnStageAble == True and self.team.firstOrder.OnStage == False and self.team.firstOrder !=self.team.battling:
                battle_able.append(0)
                battle_able_list.append(self.team.firstOrder)
            if self.team.secondOrder.OnStageAble == True and self.team.secondOrder.OnStage == False and self.team.secondOrder !=self.team.battling:
                battle_able.append(1)
                battle_able_list.append(self.team.secondOrder)
            if self.team.thirdOrder.OnStageAble == True and self.team.thirdOrder.OnStage == False and self.team.thirdOrder !=self.team.battling:
                battle_able.append(2)
                battle_able_list.append(self.team.thirdOrder)


            #先隨機挑角色上場，機制後續須補上
            #如果沒有可使用角色，則隊伍fault
            if not battle_able:
                self.team.fault = True
                #print("Team "+ f"{self.team.TeamID}"+" 全部角色皆陣亡！")
            else:
                s = battle_able[Switch_poke_decide(battle_able_list, other.team.battling)]
            
                if self.team.firstOrder.name == self.team.battling.name:
                    self.team.firstOrder.OnStage = False
                    self.team.firstOrder.HP = self.team.battling.HP
                    self.team.firstOrder.pk_energy = self.team.battling.pk_energy
                    self.team.firstOrder.OnStageAble = True
                    self.team.firstOrder.move_buff_target = None
                    self.team.firstOrder.move_atk_buff = 0
                    self.team.firstOrder.move_def_buff = 0
                if self.team.secondOrder.name == self.team.battling.name:
                    self.team.secondOrder.OnStage = False
                    self.team.secondOrder.HP = self.team.battling.HP
                    self.team.secondOrder.pk_energy = self.team.battling.pk_energy
                    self.team.secondOrder.OnStageAble = True
                    self.team.secondOrder.move_buff_target = None
                    self.team.secondOrder.move_atk_buff = 0
                    self.team.secondOrder.move_def_buff = 0
                if self.team.thirdOrder.name == self.team.battling.name:
                    self.team.thirdOrder.OnStage = False
                    self.team.thirdOrder.HP = self.team.battling.HP
                    self.team.thirdOrder.pk_energy = self.team.battling.pk_energy
                    self.team.thirdOrder.OnStageAble = True
                    self.team.thirdOrder.move_buff_target = None
                    self.team.thirdOrder.move_atk_buff = 0
                    self.team.thirdOrder.move_def_buff = 0

                if s == 0:
                    self.team.battling = self.team.firstOrder
                    self.team.firstOrder.OnStage = True
                if s == 1:
                    self.team.battling = self.team.secondOrder
                    self.team.secondOrder.OnStage = True
                if s == 2:
                    self.team.battling = self.team.thirdOrder
                    self.team.thirdOrder.OnStage = True
            self.WAIT = 0
            self.team.countT -= 1
            self.team.switchClock = 0 
            #print('換上 '+f'{self.team.battling}')
        elif self.team.countT-1 == time_slot:
            if action == 0:
                self.WAIT = 0
                self.EE, other.DD = self.team.fast_move(other.team)
            elif action == 1:
                self.WAIT = 0
                self.EE, other.DD, self.move_atk_buff, self.move_def_buff, self.move_buff_target = self.team.charged_move1(other.team)
                other.team.receivedCM = True
                other.team.countT = self.team.countT
                #print("Team "+ f"{self.team.TeamID}"+" 使用大招一, 造成傷害："+f'{other.DD}')
            elif action == 2:
                self.WAIT = 0
                self.EE, other.DD, self.move_atk_buff, self.move_def_buff, self.move_buff_target = self.team.charged_move2(other.team)
                other.team.receivedCM = True
                other.team.countT = self.team.countT
                #print("Team "+ f"{self.team.TeamID}"+" 使用大招二, 造成傷害："+f"{other.DD}")
 
        if self.team.receivedCM == True and self.team.shield >= 1 and self.team.countT == time_slot:
            if action == 5:
                other.DD = 1
                self.team.shield -= 1
                
                #print("Team "+ f"{self.team.TeamID}"+" 開盾擋下了攻擊！")
            self.team.receivedCM = False
        elif self.team.receivedCM == True and self.team.shield ==0 and self.team.countT == time_slot:
            self.team.receivedCM = False

        if time_slot == self.team.countT:
            self.team.battling.pk_energy += self.EE
            if self.team.battling.pk_energy >= 100:
                self.team.battling.pk_energy = 100
            other.team.battling.HP -= other.DD
            

            self.team.switchClock += self.WAIT
            other.team.switchClock += self.WAIT
            self.team.receivedCM = False
            other.DD = 0
            self.EE = 0
            self.WAIT = 0

            if self.move_buff_target == 'self':
                #reset atk and def to non-buff state
                if self.team.battling.atkBuffLv != 0:
                    self.team.battling.Atk /= buff_stage.get(str(self.team.battling.atkBuffLv))
                if self.team.battling.defBuffLv != 0:
                    self.team.battling.Def /= buff_stage.get(str(self.team.battling.defBuffLv))
                #new buff state
                self.team.battling.atkBuffLv += self.move_atk_buff
                self.team.battling.defBuffLv += self.move_def_buff
                if self.team.battling.atkBuffLv > 4: self.team.battling.atkBuffLv = 4
                if self.team.battling.atkBuffLv < -4: self.team.battling.atkBuffLv = -4
                if self.team.battling.defBuffLv > 4: self.team.battling.defBuffLv = 4
                if self.team.battling.defBuffLv < -4: self.team.battling.defBuffLv = -4
                #update atk and def
                self.team.battling.Atk *= buff_stage.get(str(self.team.battling.atkBuffLv))
                self.team.battling.Def *= buff_stage.get(str(self.team.battling.defBuffLv))

            if self.move_buff_target == 'opponent':
                #reset atk and def to non-buff state
                if other.team.battling.atkBuffLv != 0:
                    other.team.battling.Atk /= buff_stage.get(str(other.team.battling.atkBuffLv))
                if other.team.battling.defBuffLv != 0:
                    other.team.battling.Def /= buff_stage.get(str(other.team.battling.defBuffLv))
                #new buff state
                other.team.battling.atkBuffLv += self.move_atk_buff
                other.team.battling.defBuffLv += self.move_def_buff
                if other.team.battling.atkBuffLv > 4: other.team.battling.atkBuffLv = 4
                if other.team.battling.atkBuffLv < -4: other.team.battling.atkBuffLv = -4
                if other.team.battling.defBuffLv > 4: other.team.battling.defBuffLv = 4
                if other.team.battling.defBuffLv < -4: other.team.battling.defBuffLv = -4
                #update atk and def
                other.team.battling.Atk *= buff_stage.get(str(other.team.battling.atkBuffLv))
                other.team.battling.Def *= buff_stage.get(str(other.team.battling.defBuffLv))

            self.move_buff_target = None
            self.move_atk_buff = 0
            self.move_def_buff = 0


        

        # 己方屬性是否劣勢，是則PK_TYPE_REWARD>0
        if self.team.fault != True and other.team.fault != True:
            for i in range(0,2):
                for j in range(0,2):
                    PK_slf *=  ttype[f'{other.team.battling.type[i]}'][f'{self.team.battling.type[j]}']
                    PK_slf_oppfm *=  ttype[f'{other.team.battling.pk_fm_type}'][f'{self.team.battling.type[j]}']
                    PK_slf_oppcm1 *=  ttype[f'{other.team.battling.pk_cm1_type}'][f'{self.team.battling.type[j]}']
                    PK_slf_oppcm2 *=  ttype[f'{other.team.battling.pk_cm2_type}'][f'{self.team.battling.type[j]}']
                    

            # 敵方屬性是否劣勢，是則PK_TYPE_REWARD>0
            for i in range(0,2):
                for j in range(0,2):
                    PK_opp *=  ttype[f'{self.team.battling.type[i]}'][f'{other.team.battling.type[j]}']
                    PK_opp_slffm *=  ttype[f'{self.team.battling.pk_fm_type}'][f'{other.team.battling.type[j]}']
                    PK_opp_slfcm1 *=  ttype[f'{self.team.battling.pk_cm1_type}'][f'{other.team.battling.type[j]}']
                    PK_opp_slfcm2 *=  ttype[f'{self.team.battling.pk_cm2_type}'][f'{other.team.battling.type[j]}']
            
            next_state = [PK_opp, PK_slf]
            if PK_opp/PK_slf >= 1:
                reward *= PK_opp/PK_slf/3
            else:
                reward *= -1*PK_slf/PK_opp/3
            reward -= (PK_slf_oppfm + PK_slf_oppcm1 + PK_slf_oppcm2)
            reward += (PK_opp_slffm + PK_opp_slfcm1 + PK_opp_slfcm2)

            with open('Pokemon_'+f'{self.team.CP_limit}'+'_default.json', 'r', encoding="utf-8") as json_file:
                all_data = json.load(json_file)

            reward += (self.team.battling.HP/all_data[f'{self.team.battling.name}']['CurrentHP'] - other.team.battling.HP/all_data[f'{other.team.battling.name}']['CurrentHP'])*2
            
            reward += (self.team.battling.pk_energy/100 - other.team.battling.pk_energy/100)*2
            if other.team.battling.atkBuffLv < 0: reward -= 1
            if other.team.battling.defBuffLv < 0: reward -= 1
            if self.team.battling.atkBuffLv > 0: reward += 1
            if self.team.battling.defBuffLv > 0: reward += 1
            next_state = reward
            return next_state, reward, done, action
        else: 
            return reward, reward, done, action

class DQNAgent:
    def __init__(self, model=None):
        # 初始化 DQN 代理
        if model is not None:
            self.model = model
            self.target_model = model
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()

        
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # 記憶體，存儲先前的環境資訊

        # Custom tensorboard object
        #self.tensorboard = ModifiedTensorBoard(log_dir="PokeDQN_logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        self.learning_rate = 0.001  # 模型的學習率
        self.epsilon = 1.0 
        
        self.gamma = 0.95  # 折扣因子，控制未來獎勵的重要性
        '''
         # 探索率，隨機選擇行動的概率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率的衰減率
        '''
        

    def create_model(self):
        # 構建深度神經網路模型
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(1,)))  # 請替換 your_input_dim 為實際的輸入維度
        model.add(Dense(128, activation='relu'))
        model.add(Dense(your_env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=adam_v2.Adam(lr=0.001))

        return model

    def update_replay_memory(self, transition):
        #transition裡面有(?)(state, action, reward, next_state, done)
        # 將環境資訊存入記憶體
        
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        # 從記憶體中取樣一個小批次進行 Q-learning 更新模型
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_states = np.asarray([transition[0] for transition in minibatch], dtype="object").astype('float32')
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.asarray([transition[3] for transition in minibatch], dtype="object").astype('float32')
        future_qs_list = self.target_model.predict(new_current_states)

        #current_state
        X = []
        #Q VALUE
        Y = []
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = (reward + self.gamma * max_future_q)
            else:
                new_q = reward
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(state)
            Y.append(current_qs)
            
        self.model.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 隨時間降低探索率
        '''
    # Queries main network for Q values given current observation space (environment state)
        
    def get_qs(self, state):
        return self.model.predict(np.asarray([state], dtype="object").astype('float32'))[0]

class DQNAgentTwo:
    def __init__(self, model=None):
        # 初始化 DQN 代理
        if model is not None:
            self.model = model
            self.target_model = model
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()

        
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # 記憶體，存儲先前的環境資訊

        # Custom tensorboard object
        #self.tensorboard = ModifiedTensorBoard(log_dir="PokeDQN_logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        self.learning_rate = 0.001  # 模型的學習率
        self.epsilon = 1.0 
        
        self.gamma = 0.95  # 折扣因子，控制未來獎勵的重要性
        '''
         # 探索率，隨機選擇行動的概率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率的衰減率
        '''
        

    def create_model(self):
        # 構建深度神經網路模型
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(1,)))  # 請替換 your_input_dim 為實際的輸入維度
        model.add(Dense(128, activation='relu'))
        model.add(Dense(opp_env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=adam_v2.Adam(lr=0.001))
        return model

    def update_replay_memory(self, transition):
        #transition裡面有(?)(state, action, reward, next_state, done)
        # 將環境資訊存入記憶體
        
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        # 從記憶體中取樣一個小批次進行 Q-learning 更新模型
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_states = np.asarray([transition[0] for transition in minibatch], dtype="object").astype('float32')
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.asarray([transition[3] for transition in minibatch], dtype="object").astype('float32')
        future_qs_list = self.target_model.predict(new_current_states)

        #current_state
        X = []
        #Q VALUE
        Y = []
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = (reward + self.gamma * max_future_q)
            else:
                new_q = reward
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(state)
            Y.append(current_qs)
            
        self.model.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 隨時間降低探索率
        '''
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.asarray([state], dtype="object").astype('float32'))[0]



# 初始化模型、Agent、Env等
set_session(sess)    
your_model = load_model('models/PokeBattle___537.59max__283.32avg_2000episodes__Fri_Jan_19_08_51_19_2024.model')
your_agent = DQNAgent(model=your_model)
your_env = PokemonBattleEnv()
set_session(sess2)
opp_model = load_model('models/PokeBattle___537.59max__283.32avg_2000episodes__Fri_Jan_19_08_51_19_2024.model')
opp_agent = DQNAgentTwo(model = opp_model)
opp_env = PokemonBattleEnvTwo()

T1 = [["妙蛙花","VINE_WHIP", "FRENZY_PLANT", "SLUDGE_BOMB"],
            ["巨沼怪","MUD_SHOT", "HYDRO_CANNON", "EARTHQUAKE"],
            ["白海獅","ICE_SHARD", "ICY_WIND", "DRILL_RUN"]]
T2 = [["盔甲鳥","STEEL_WING", "SKY_ATTACK", "BRAVE_BIRD"],
        ["巨沼怪","MUD_SHOT", "HYDRO_CANNON", "EARTHQUAKE"],
        ["負電拍拍","QUICK_ATTACK", "DISCHARGE", "GRASS_KNOT"]]

Team1 = Team(T1,1,1500)
Team2 = TeamTwo(T2,2,1500)

with open('default_1500_team.json', 'r', encoding="utf-8") as defaultTeam:
    dteam = json.load(defaultTeam)
teamcount = 80

Team2.reset(dteam[teamcount],2,1500)
Team1.reset(T1,1,1500)
    
state = your_env.reset(Team1)  # 獲得初始狀態
state2 = opp_env.reset(Team2)  # 獲得初始狀態

set_session(sess)
s2= your_model.predict(np.asarray([state], dtype="object").astype('float32'))[0]
                
set_session(sess2)
s2= opp_model.predict(np.asarray([state2], dtype="object").astype('float32'))[0]
@app.route('/')
def home():
    
    # 在這裡使用 BattleSimulator 進行模擬對戰
    
    
    num_battles = 1  # 可調整
    wins = 0
    draws = 0
    losses = 0
    teamcount = 0
    for episode in range(num_battles):
        time_slot = 270*2
        episode_reward = 0
        with open('default_1500_team.json', 'r', encoding="utf-8") as defaultTeam:
            dteam = json.load(defaultTeam)
        teamcount = 80+int(episode)
        Team2.reset(dteam[teamcount],2,1500)
        Team1.reset(T1,1,1500)
            
        state = your_env.reset(Team1)  # 獲得初始狀態
        state2 = opp_env.reset(Team2)  # 獲得初始狀態

        #state = np.reshape(state, [1, env.OBSERVATION_SPACE_VALUES])
        done = False

        while not done:
            #app.logger.info('第 '+str(540-time_slot)+' 回合, 隊伍一' +f'{env.team.battling.name}'+'剩餘血量:'f'{env.team.battling.HP}' + ', 隊伍二' +f'{opp_env.team.battling.name}'+'剩餘血量:'f'{opp_env.team.battling.HP}')
            time_slot -= 1 
            your_env.team.switchClock += 1
            opp_env.team.switchClock += 1
            down2_count = 0
            if opp_env.team.firstOrder.OnStageAble == False:
                down2_count += 1
            if opp_env.team.secondOrder.OnStageAble == False:
                down2_count += 1
            if opp_env.team.thirdOrder.OnStageAble == False:
                down2_count += 1

            if down2_count >= 2:
                opp_env.switching = False

            down_count = 0
            if your_env.team.firstOrder.OnStageAble == False:
                down_count += 1
            if your_env.team.secondOrder.OnStageAble == False:
                down_count += 1
            if your_env.team.thirdOrder.OnStageAble == False:
                down_count += 1

            if down_count >= 2:
                your_env.switching = False

            if time_slot > 0 and your_env.team.fault != True and opp_env.team.fault != True:
                allowed2 = False    
                global opp_model
                global sess2
                global graph2
                with graph2.as_default():
                    set_session(sess2)
                    s2= opp_model.predict(np.asarray([state2], dtype="object").astype('float32'))[0]
                while not allowed2:      
                    action = np.argmax(s2)
                    if opp_env.team.receivedCM == False:
                        if action  == 0 and opp_env.team.battling.pk_energy < 100:
                            allowed2 = True
                        elif action  == 1 and opp_env.team.battling.pk_energy >= -1*opp_env.team.battling.pk_cm1_energyLoss:
                            allowed2 = True
                        elif action  == 2 and opp_env.team.battling.pk_energy >= -1*opp_env.team.battling.pk_cm2_energyLoss:
                            allowed2 = True
                        elif action  == 3:
                            if random.random()< 0.5:
                                allowed2 = False
                            else:
                                allowed2 = False
                        elif action  == 4 and opp_env.team.switchClock >= 60*2 and opp_env.switching == True:
                            allowed2 = True
                    elif opp_env.team.receivedCM == True:
                        if opp_env.team.shield > 0:
                            if action == 5:
                                allowed2 = True
                            elif action == 4 and opp_env.team.switchClock >= 60*2 and opp_env.switching == True:
                                allowed2 = True
                            elif action == 3:
                                allowed2 = True
                        else: 
                            if action == 4:
                                allowed2 = True
                            elif action == 3:
                                allowed2 = True  
                    s2[np.argmax(s2)] = s2[np.argmin(s2)] -1
                
                next_state2, reward2, done2, ac = opp_env.step(your_env,action)  # 執行行動並獲得下一個狀態、獎勵和終止條件
                opp_agent.update_replay_memory((state2, action, reward2, next_state2, done))  # 存儲環境資訊
                opp_agent.train(done2)
                state2 = next_state2  # 更新狀態
    
                allowed = False 
                global your_model
                global sess
                global graph
                with graph.as_default():
                    set_session(sess)   
                    s= your_model.predict(np.asarray([state], dtype="object").astype('float32'))[0]
                while not allowed:      
                    action = np.argmax(s)
                    if your_env.team.receivedCM == False:
                        if action  == 0 and your_env.team.battling.pk_energy < 100:
                            allowed = True
                        elif action  == 1 and your_env.team.battling.pk_energy >= -1*your_env.team.battling.pk_cm1_energyLoss:
                            allowed = True
                        elif action  == 2 and your_env.team.battling.pk_energy >= -1*your_env.team.battling.pk_cm2_energyLoss:
                            allowed = True
                        elif action  == 3:
                            if random.random()< 0.5:
                                allowed = False
                            else:
                                allowed = False
                        elif action  == 4 and your_env.team.switchClock >= 60*2 and your_env.switching == True:
                            allowed = True
                    elif your_env.team.receivedCM == True:
                        if your_env.team.shield > 0:
                            if action == 5:
                                allowed = True
                            elif action == 4:
                                allowed = True
                            elif action == 3:
                                allowed = True
                        else: 
                            if action == 4 and your_env.team.switchClock >= 60*2 and your_env.switching == True:
                                allowed = True
                            elif action == 3:
                                allowed = True  
                        
                    s[np.argmax(s)] = s[np.argmin(s)] -1
                app.logger.info(f"{your_env.team.switchClock}")
                next_state, reward, done, ac = your_env.step(opp_env,action)  # 執行行動並獲得下一個狀態、獎勵和終止條件
                episode_reward += reward

                your_agent.update_replay_memory((state, action, reward, next_state, done))  # 存儲環境資訊
                your_agent.train(done)
                state = next_state  # 更新狀態
            

            if time_slot == 0:
                app.logger.info("Draw")
                app.logger.info('隊伍一' +f'{your_env.team.battling.name}'+'剩餘血量:'f'{your_env.team.battling.HP}' + ', 隊伍二' +f'{opp_env.team.battling.name}'+'剩餘血量:'f'{opp_env.team.battling.HP}')
                result = 'Draw'
                done = True
            elif your_env.team.fault == True and opp_env.team.fault == True:
                app.logger.info("Team "+f"{your_env.team.TeamID}"+"and Team "+f"{opp_env.team.TeamID}"+" down at the same time")
                result = 'Draw'
                done = True
            elif your_env.team.fault == True:
                app.logger.info("Team "+f"{your_env.team.TeamID}"+" Lose")
                result = 'lose'
                done = True
            elif opp_env.team.fault == True:
                app.logger.info("Oppenent Team "+f"{opp_env.team.TeamID}"+" Lose")
                result = 'Win'
                done = True

        # 更新勝場和敗場計數    
        if result == 'Win':
            wins += 1
        elif result == 'Draw':
            draws += 1
        else:
            losses += 1
        app.logger.info(str(episode)+' end')

    # 輸出勝率
    win_rate = wins / num_battles
    app.logger.info(f'勝率: {win_rate * 100}%')
    return 'hello'

if __name__ == '__main__':
    app.run(debug=True)