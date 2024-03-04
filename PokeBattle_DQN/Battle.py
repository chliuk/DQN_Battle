from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
import math
import json


def energy_decrease(power, currentAtk, currentDef, stab, resist):
    return math.floor(0.5*power*currentAtk/currentDef*stab*resist)+1

def Damage(atkpk_power,atkpk_ATK,defpk_DEF):
    return math.floor(0.5*atkpk_power*atkpk_ATK/defpk_DEF*1.3)+1

STAB = 1.2

Neutral = 1
SuperEffective = 1.6
Not_Very_Effective = 0.625
Immune = 0.390625

def battle(pk1, pk1_fm, pk1_cm1, pk1_cm2, pk2, pk2_fm, pk2_cm1, pk2_cm2):
    with open('moves.json', 'r') as m:
        mov = json.load(m)

    with open('Pokemon_1500_default.json', 'r', encoding="utf-8") as json_file:
        all_data = json.load(json_file)

    with open('type.json', 'r', encoding="utf-8") as ty:
        ttype = json.load(ty)

    for move_data in mov:
        if move_data.get('moveId') == pk1_fm:
            pk1_fm_type =  move_data.get('type')
            pk1_fm_power =  move_data.get('power')
            pk1_fm_energyGain =  move_data.get('energyGain')
            pk1_fm_turn =  move_data.get('cooldown')/500
            
        elif move_data.get('moveId') == pk2_fm:
            pk2_fm_type =  move_data.get('type')
            pk2_fm_power =  move_data.get('power')
            pk2_fm_energyGain =  move_data.get('energyGain')
            pk2_fm_turn =  move_data.get('cooldown')/500
            print(pk2_fm_power)
        elif move_data.get('moveId') == pk1_cm1:
            pk1_cm1_type =  move_data.get('type')
            pk1_cm1_power =  move_data.get('power')
            pk1_cm1_energyLoss =  -1*move_data.get('energy')
            pk1_cm1_turn =  1
        elif move_data.get('moveId') == pk1_cm2:
            pk1_cm2_type =  move_data.get('type')
            pk1_cm2_power =  move_data.get('power')
            pk1_cm2_energyLoss =  -1*move_data.get('energy')
            pk1_cm2_turn =  1
        elif move_data.get('moveId') == pk2_cm1:
            pk2_cm1_type =  move_data.get('type')
            pk2_cm1_power =  move_data.get('power')
            pk2_cm1_energyLoss =  -1*move_data.get('energy')
            pk2_cm1_turn =  1
        elif move_data.get('moveId') == pk2_cm2:
            pk2_cm2_type =  move_data.get('type')
            pk2_cm2_power =  move_data.get('power')
            pk2_cm2_energyLoss =  -1*move_data.get('energy')
            pk2_cm2_turn =  1

    [pk1_type1, pk1_type2]= all_data[pk1]['type']
    [pk2_type1, pk2_type2]= all_data[pk2]['type']

    #算pk1的屬修
    if pk1_fm_type == pk1_type1:
        pk1_fm_power *= STAB
    if pk1_fm_type == pk1_type2:
        pk1_fm_power *= STAB

    if pk1_cm1_type == pk1_type1:
        pk1_cm1_power *= STAB
    if pk1_cm1_type == pk1_type2:
        pk1_cm1_power *= STAB

    if pk1_cm2_type == pk1_type1:
        pk1_cm2_power *= STAB
    if pk1_cm2_type == pk1_type2:
        pk1_cm2_power *= STAB

    #算pk2的屬修
    if pk2_fm_type == pk2_type1:
        pk2_fm_power *= STAB
    if pk2_fm_type == pk2_type2:
        pk2_fm_power *= STAB

    if pk2_cm1_type == pk2_type1:
        pk2_cm1_power *= STAB
    if pk2_cm1_type == pk2_type2:
        pk2_cm1_power *= STAB

    if pk2_cm2_type == pk2_type1:
        pk2_cm2_power *= STAB
    if pk2_cm2_type == pk2_type2:
        pk2_cm2_power *= STAB


    #算pk1的屬性相剋加成
    if ttype[pk1_fm_type][pk2_type1] == 1.6:
        pk1_fm_power *= SuperEffective
    elif ttype[pk1_fm_type][pk2_type1] == 0.625:
        pk1_fm_power *= Not_Very_Effective
    elif ttype[pk1_fm_type][pk2_type1] == 0.391:
        pk1_fm_power *= Immune 

    if ttype[pk1_fm_type][pk2_type2] == 1.6:
        pk1_fm_power *= SuperEffective
    elif ttype[pk1_fm_type][pk2_type2] == 0.625:
        pk1_fm_power *= Not_Very_Effective
    elif ttype[pk1_fm_type][pk2_type2] == 0.391:
        pk1_fm_power *= Immune 

    if ttype[pk1_cm1_type][pk2_type1] == 1.6:
        pk1_cm1_power *= SuperEffective
    elif ttype[pk1_cm1_type][pk2_type1] == 0.625:
        pk1_cm1_power *= Not_Very_Effective
    elif ttype[pk1_cm1_type][pk2_type1] == 0.391:
        pk1_cm1_power *= Immune 

    if ttype[pk1_cm1_type][pk2_type2] == 1.6:
        pk1_cm1_power *= SuperEffective
    elif ttype[pk1_cm1_type][pk2_type2] == 0.625:
        pk1_cm1_power *= Not_Very_Effective
    elif ttype[pk1_cm1_type][pk2_type2] == 0.391:
        pk1_cm1_power *= Immune 

    if ttype[pk1_cm2_type][pk2_type1] == 1.6:
        pk1_cm2_power *= SuperEffective
    elif ttype[pk1_cm2_type][pk2_type1] == 0.625:
        pk1_cm2_power *= Not_Very_Effective
    elif ttype[pk1_cm2_type][pk2_type1] == 0.391:
        pk1_cm2_power *= Immune 

    if ttype[pk1_cm2_type][pk2_type2] == 1.6:
        pk1_cm2_power *= SuperEffective
    elif ttype[pk1_cm2_type][pk2_type2] == 0.625:
        pk1_cm2_power *= Not_Very_Effective
    elif ttype[pk1_cm2_type][pk2_type2] == 0.391:
        pk1_cm2_power *= Immune 
    
    #算pk2的屬性相剋加成
    if ttype[pk2_fm_type][pk1_type1] == 1.6:
        pk2_fm_power *= SuperEffective
    elif ttype[pk2_fm_type][pk1_type1] == 0.625:
        pk2_fm_power *= Not_Very_Effective
    elif ttype[pk2_fm_type][pk1_type1] == 0.391:
        pk2_fm_power *= Immune 

    if ttype[pk2_fm_type][pk1_type2] == 1.6:
        pk2_fm_power *= SuperEffective
    elif ttype[pk2_fm_type][pk1_type2] == 0.625:
        pk2_fm_power *= Not_Very_Effective
    elif ttype[pk2_fm_type][pk1_type2] == 0.391:
        pk2_fm_power *= Immune 

    if ttype[pk2_cm1_type][pk1_type1] == 1.6:
        pk2_cm1_power *= SuperEffective
    elif ttype[pk2_cm1_type][pk1_type1] == 0.625:
        pk2_cm1_power *= Not_Very_Effective
    elif ttype[pk2_cm1_type][pk1_type1] == 0.391:
        pk2_cm1_power *= Immune 

    if ttype[pk2_cm1_type][pk1_type2] == 1.6:
        pk2_cm1_power *= SuperEffective
    elif ttype[pk2_cm1_type][pk1_type2] == 0.625:
        pk2_cm1_power *= Not_Very_Effective
    elif ttype[pk2_cm1_type][pk1_type2] == 0.391:
        pk2_cm1_power *= Immune 

    if ttype[pk2_cm2_type][pk1_type1] == 1.6:
        pk1_cm2_power *= SuperEffective
    elif ttype[pk2_cm2_type][pk1_type1] == 0.625:
        pk1_cm2_power *= Not_Very_Effective
    elif ttype[pk2_cm2_type][pk1_type1] == 0.391:
        pk1_cm2_power *= Immune 

    if ttype[pk2_cm2_type][pk1_type2] == 1.6:
        pk2_cm2_power *= SuperEffective
    elif ttype[pk2_cm2_type][pk1_type2] == 0.625:
        pk2_cm2_power *= Not_Very_Effective
    elif ttype[pk2_cm2_type][pk1_type2] == 0.391:
        pk2_cm2_power *= Immune 

    pk1_energy = 0;
    pk1_ATK = all_data[pk1]['CurrentAtk']
    pk1_DEF = all_data[pk1]['CurrentDef']
    pk1_HP = int(all_data[pk1]['CurrentHP'])

    pk2_energy = 0;
    pk2_ATK = all_data[pk2]['CurrentAtk']
    pk2_DEF = all_data[pk2]['CurrentDef']
    pk2_HP = int(all_data[pk2]['CurrentHP'])

    time_slot = 270*2
    countpk1 = time_slot
    countpk2 = time_slot
    
    for t in range(time_slot,0,-1):
        if countpk1 == t:
            #if pk1_energy < min(pk1_cm1_energyLoss,pk1_cm2_energyLoss):
            pk1_energy += pk1_fm_energyGain
            pk2_HP -= Damage(pk1_fm_power,pk1_ATK,pk2_DEF)
            countpk1 -= pk1_fm_turn
            #elif pk1_energy > min(pk1_cm1_energyLoss,pk1_cm2_energyLoss) and pk1_energy < max(pk1_cm1_energyLoss,pk1_cm2_energyLoss):
            #   if pk1_cm1_type_flag == True:
            #        #出大招
            #        Damage()
            #    else:
            #        Damage()
                
        if countpk2 == t:
            pk2_energy += pk2_fm_energyGain
            pk1_HP -= Damage(pk2_fm_power,pk2_ATK,pk1_DEF)
            countpk2 -= pk2_fm_turn
    
        if min(pk1_HP,pk2_HP) <= 0:
            print("Winner is "+ pk1 + ", 剩餘血量 " + str(max(pk1_HP,pk2_HP)))
            break
        
battle("妙蛙花","VINE_WHIP", "FRENZY_PLANT", "SLUDGE_BOMB", "電燈怪","SPARK","SURF","THUNDERBOLT")

