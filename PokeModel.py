from keras.models import load_model
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

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
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

STAB = 1.2

SuperEffective = 1.6
Not_Very_Effective = 0.625
Immune = 0.390625

time_slot = 270*2

def Damage(atkpk_power,atkpk_ATK,defpk_DEF):
    return math.floor(0.5*atkpk_power*atkpk_ATK/defpk_DEF*1.3)+1

class Team:
    def __init__(self, pkList, teamID, CP_limit):
        self.countT = time_slot
        self.firstOrder = Pokemon(pkList[0][0], pkList[0][1], pkList[0][2], pkList[0][3], CP_limit)
        self.secondOrder = Pokemon(pkList[1][0], pkList[1][1], pkList[1][2], pkList[1][3], CP_limit)
        self.thirdOrder = Pokemon(pkList[2][0], pkList[2][1], pkList[2][2], pkList[2][3], CP_limit)
        self.TeamID = teamID
        self.firstOrder.OnStage = True
        self.battling = self.firstOrder
        self.battling.last_action_damage = 0
        self.shield = 2
        self.switchClock = 0
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
        self.battling = self.firstOrder
        self.battling.last_action_damage = 0
        self.shield = 2
        self.switchClock = 0
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
        #不確定other, self的用法
        pokemon_list = [self.firstOrder, self.secondOrder, self.thirdOrder] 
        self.switchClock+=1
        if self.battling.HP <= 0:
            battle_able = []
            for pk in pokemon_list:
                if pk.OnStageAble == True and pk.OnStage == False:
                    battle_able = [battle_able, pk]

            #先隨機挑角色上場，機制後續須補上
            if not battle_able:
                self.fault = True
                #print("沒有可使用的寶可夢")
            else:
                if len(battle_able) == 1:                    
                    self.switch_teammate(self.battling, battle_able[0])
                    self.battling = battle_able[0]
                else:
                    if random.uniform(0, 1) >=0.5:                        
                        self.switch_teammate(self.battling, battle_able[0])
                        self.battling = battle_able[0]
                    else:                        
                        self.switch_teammate(self.battling, battle_able[1])    
                        self.battling = battle_able[1]

            waste_time = random.randint(5, 20)
            return waste_time
        
        #waste_time = max(waste_time,waste_time)
        #timeslot -= waste_time
        #other.countT = timeslot
        #other.switchClock -= waste_time
        #self.countT = timeslot
        #self.switchClock -= waste_time


        if self.receivedCM == False:
            if self.battling.HP >= 0:
                if choice == 0:
                    self.fast_move(other)
                elif choice == 1 and self.battling.pk_energy >= -1*self.battling.pk_cm1_energyLoss:
                    self.charged_move1(other)
                elif choice == 2 and self.battling.pk_energy >= -1*self.battling.pk_cm1_energyLoss:
                    self.charged_move2(other)
                elif choice == 3:
                    self.countT -= 1
                    #print("不動作一回合")
                elif choice == 4 and self.switchClock >= 60:
                    battle_able = []
                    for pk in pokemon_list:
                        if pk.OnStageAble == True and pk.OnStage == False:
                            battle_able = battle_able + [pk]

                    #先隨機挑角色上場，機制後續須補上
                    if not battle_able:
                        self.fault = True
                        #print("沒有可使用的寶可夢")
                    else:
                        if len(battle_able) == 1:                    
                            self.switch_teammate(self.battling, battle_able[0])
                            self.battling = battle_able[0]
                            self.switchClock = 0
                        else:
                            if random.uniform(0, 1) >=0.5:                        
                                self.switch_teammate(self.battling, battle_able[0])
                                self.battling = battle_able[0]
                                self.switchClock = 0
                                
                            else:                        
                                self.switch_teammate(self.battling, battle_able[1])    
                                self.battling = battle_able[1]  
                                self.switchClock = 0
                                    
        else:
            if self.countT == time_slot and self.switchClock >= 60:
                if choice == 4:
                    battle_able = []
                    for pk in pokemon_list:
                        if pk.OnStageAble == True and pk.OnStage == False:
                            battle_able = battle_able + [pk]
                            

                    #先隨機挑角色上場，機制後續須補上
                    if not battle_able:
                        self.fault = True
                        #print("沒有可使用的寶可夢")
                    else:
                        if len(battle_able) == 1:                    
                            self.switch_teammate(self.battling, battle_able[0])
                            self.battling = battle_able[0]
                            self.switchClock = 0
                        else:
                            if random.uniform(0, 1) >=0.5:                        
                                self.switch_teammate(self.battling, battle_able[0])
                                self.battling = battle_able[0]
                                self.switchClock = 0
                            else:                      
                                self.switch_teammate(self.battling, battle_able[1])    
                                self.battling = battle_able[1]  
                                self.switchClock = 0


            else:        
                if choice == 5 and self.shield > 0:
                    self.shield -= 1
                    self.battling.HP += other.battling.last_action_damage-1
                    


    def fast_move(self, other):
        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)

        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_fm_power *= SuperEffective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_fm_power *= Not_Very_Effective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_fm_power *= Immune

        if ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_fm_power *= SuperEffective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_fm_power *= Not_Very_Effective
        elif ttype[self.battling.pk_fm_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_fm_power *= Immune
        
        self.battling.pk_energy += self.battling.pk_fm_energyGain
        if  self.battling.pk_energy >= 100:
            self.battling.pk_energy = 100
        D = Damage(self.battling.pk_fm_power,self.battling.Atk, other.battling.Def)
        other.battling.HP -= D
        other.battling.last_action_damage = D
        self.countT -= self.battling.pk_fm_turn
        return D
    
    def charged_move1(self, other):
        other.receivedCM = True

        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)

        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_cm1_power *= SuperEffective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_cm1_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_cm1_power *= Immune

        if ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_cm1_power *= SuperEffective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_cm1_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm1_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_cm1_power *= Immune

        self.battling.pk_energy += self.battling.pk_cm1_energyLoss
        D = Damage(self.battling.pk_cm1_power,self.battling.Atk, other.battling.Def)
        other.battling.HP -= D
        self.countT -= self.battling.pk_cm1_turn
        return D

    def charged_move2(self, other):
        other.receivedCM = True

        with open('type.json', 'r', encoding="utf-8") as ty:
            ttype = json.load(ty)

        #計算屬性相剋傷害加成
        if ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 1.6:
            self.battling.pk_cm2_power *= SuperEffective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 0.625:
            self.battling.pk_cm2_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type1] == 0.391:
            self.battling.pk_cm2_power *= Immune

        if ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 1.6:
            self.battling.pk_cm2_power *= SuperEffective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 0.625:
            self.battling.pk_cm2_power *= Not_Very_Effective
        elif ttype[self.battling.pk_cm2_type][other.battling.pk_type2] == 0.391:
            self.battling.pk_cm2_power *= Immune

        self.battling.pk_energy += self.battling.pk_cm2_energyLoss
        D = Damage(self.battling.pk_cm2_power,self.battling.Atk, other.battling.Def)
        other.battling.HP -= D
        self.countT -= self.battling.pk_cm2_turn
        return D 
    
    def switch_teammate(self, down, up):       
        if down in [self.firstOrder, self.secondOrder, self.thirdOrder]:
            if down.HP <= 0:
                down.OnStage = False
                down.OnStageAble = False
            else:
                down.OnStage = False

            if up in [self.firstOrder, self.secondOrder, self.thirdOrder] and up.OnStageAble == True:
                # 將上場的 Pokemon 進入戰鬥
                up.OnStage = True
                #print(f"{up.name} 上場")
                return 1
            else:           
                self.fault = True  
                print("Team "+ f"{self.TeamID}"+" 全部角色皆陣亡！")
                return 0
        else:
            print(f"不屬於隊伍的角色！")

class Pokemon:
    def __init__(self, name, pk_fm, pk_cm1, pk_cm2, CP_limit):
        self.name = name

        self.TeamID = 0
        self.OnStage = False
        self.OnStageAble = True

        self.last_action_damage = 0

        with open('Pokemon_'+str(CP_limit)+'_default.json', 'r', encoding="utf-8") as json_file:
            all_data = json.load(json_file) 

        self.Atk = all_data[name]['CurrentAtk']
        self.Def = all_data[name]['CurrentDef']
        self.HP= all_data[name]['CurrentHP']
        self.pk_energy = 0
        

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
        #算pk的屬修
        if self.pk_fm_type == self.pk_type1:
            self.pk_fm_power *= STAB
        if self.pk_fm_type == self.pk_type2:
            self.pk_fm_power *= STAB

        if self.pk_cm1_type == self.pk_type1:
            self.pk_cm1_power *= STAB
        if self.pk_cm1_type == self.pk_type2:
            self.pk_cm1_power *= STAB

        if self.pk_cm2_type == self.pk_type1:
            self.pk_cm2_power *= STAB
        if self.pk_cm2_type == self.pk_type2:
            self.pk_cm2_power *= STAB

class PokemonBattleEnv:

    SIZE = 10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    OBSERVATION_SPACE_VALUES = np.expand_dims(OBSERVATION_SPACE_VALUES, axis = -1)
    ACTION_SPACE_SIZE = 6

    def reset(self, team):
        self.team = team
        #self.team2 = team2
        self.team_switch_clock = 60*2
        #self.team1_switch_clock = 60*2
        self.team_shield = 2
        self.shieldusing = True
        self.switching = True
        #self.team2_shield = 2
        self.turns = time_slot
        self.team.fault = False
        return 1

    #Team的這一步做了甚麼、做得好不好
    def step(self, other, action):

        if other.team.fault == True or self.team.fault == True:
            pass
        else:
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
            self.team.action(other.team,action)
            #print(other.team.battling)
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

                reward += (self.team.battling.HP/all_data[f'{self.team.battling.name}']['CurrentHP'] - other.team.battling.HP/all_data[f'{other.team.battling.name}']['CurrentHP'])/20
                
                reward += (self.team.battling.pk_energy/100 - other.team.battling.pk_energy/100)/20
                self.turns -= 1
                # 返回狀態、獎勵、終止條件
                
                if self.turns == 0:
                    print("Draw")
                    done = True
                elif self.team.fault == True:
                    print("Team "+f"{self.team.teamID}"+" Lose")
                    done = True
                elif other.team.fault == True:
                    print("Oppenent Team "+f"{other.team.teamID}"+" Lose")
                    done = True
                
                next_state = reward
                return next_state, reward, done, action
            else:
                return reward, reward, done, action

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
   
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter
    
        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
    
        self._should_write_train_graph = False  
   
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

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
        self.tensorboard = ModifiedTensorBoard(log_dir="PokeDQN_logs/{}-{}".format(MODEL_NAME, int(time.time())))

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
        model.add(Dense(128, activation='relu'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.add(Dense(128, activation='relu'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=adam_v2.Adam(lr=0.001))
        model.build((32,1))
        #model.summary()
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
        #print(current_states)
        current_qs_list = self.model.predict(current_states)
        #print('----------')
        #print(current_qs_list)
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
        #print(self.model.predict(np.asarray([state], dtype="object").astype('float32'))[0])
        return self.model.predict(np.asarray([state], dtype="object").astype('float32'))[0]
# 載入模型

model = load_model('models/2x256____82.48max__-64.10avg_-232.01min__1703999546.model')

env = PokemonBattleEnv()
env2 = PokemonBattleEnv()

agent = DQNAgent(model= model)
agent2 = DQNAgent(model = model)

num_battles = 10

wins = 0
losses = 0
draws = 0

T2 = [["盔甲鳥","STEEL_WING", "SKY_ATTACK", "BRAVE_BIRD"],
        ["巨沼怪","MUD_SHOT", "HYDRO_CANNON", "EARTHQUAKE"],
        ["負電拍拍","QUICK_ATTACK", "DISCHARGE", "GRASS_KNOT"]]
T1 = [["妙蛙花","VINE_WHIP", "FRENZY_PLANT", "SLUDGE_BOMB"],
        ["天蠍","WING_ATTACK", "AERIAL_ACE", "DIG"],
        ["電燈怪","SPARK", "SURF", "THUNDERBOLT"]]

result = None

for i in range(num_battles):

    result = 0
    your_team = Team(T1,1,1500)
    your_team.reset(T1,1,1500)
    state = env.reset(your_team)

    with open('default_1500_team.json', 'r', encoding="utf-8") as defaultTeam:
        dteam = json.load(defaultTeam)

    default_team = Team(dteam[i],1,1500)
    default_team.reset(dteam[i],1,1500)
    env2.reset(your_team)
    done= False

    '''next_state, reward, done, ac = env.step(env2,0)
    agent.update_replay_memory((state, 0, reward, next_state, done))'''

 
    '''minibatch = random.sample(agent.replay_memory, MINIBATCH_SIZE)
 
    current_states = np.asarray([transition[0] for transition in minibatch], dtype="object").astype('float32')
    current_qs = model.predict(current_states)'''

    while not done:
        action = np.argmax(agent.get_qs(state))
        #print(action)
        if env.team.fault != True and env2.team.fault != True:
            env2.step(env,0)
        elif env.team.fault == True and env2.team.fault != True:
            result = 'Lose'
            print("Lose")
        elif env.team.fault != True and env2.team.fault == True:
            result = 'Win'
            print("Win")
        else:
            result = 'Draw'
            print("Draw")
            done = True

        if env.turns > 0 and env.team.fault != True and env2.team.fault != True:
            next_state, reward, done, ac = env.step(env2,action)  # 執行行動並獲得下一個狀態、獎勵和終止條件
            agent.update_replay_memory((state, action, reward, next_state, done))  # 存儲環境資訊
            agent.train(done)
            state = next_state
        elif env.turns == 0 and env.team.fault != True and env2.team.fault != True:
            print("Draw")
            done = True
    print(f"env.team.fault: {env.team.fault}, env2.team.fault: {env2.team.fault}")


    # 更新勝場和敗場計數
    if result == 'Win':
        wins += 1
    elif result == 'Draw':
        draws += 1
    else:
        losses += 1

# 輸出勝率
win_rate = wins / num_battles
print(f'勝率: {win_rate * 100}%')