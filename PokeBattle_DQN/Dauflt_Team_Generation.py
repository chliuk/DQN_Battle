import random
import json

# 讀取1500.json檔案
with open("Pokemon_1500_default.json", "r", encoding="utf-8") as file:
    pokemon_data = json.load(file)

# 生成對戰陣容的函數
def generate_team():
    # 隊伍範例
    example_team = [["妙蛙花", "VINE_WHIP", "FRENZY_PLANT", "SLUDGE_BOMB"],
                    ["天蠍", "WING_ATTACK", "AERIAL_ACE", "DIG"],
                    ["電燈怪", "SPARK", "SURF", "THUNDERBOLT"]]

    teams = []
    # 低CP對戰組合的百分比
    low_cp_percentage = 0.08
    num_teams = 3000

    for _ in range(num_teams):
        team = []
        for pokemon_name, moves in random.sample(pokemon_data.items(), len(example_team)):
            # 根據文件中的CP信息判斷
            if 1450 <= moves["DefaulCP"] <= 1500:
                # 隨機選擇一個快速招式
                selected_fast_move = random.choice(moves["fastMoves"])
                # 隨機選擇兩個充能招式
                selected_charged_moves = random.sample(moves["chargedMoves"], 2)
                
                team.append([pokemon_name, selected_fast_move] + selected_charged_moves)

        # 如果隊伍中包含的Pokemon數量與範例隊伍相同，則將其加入隊伍列表
        if len(team) == len(example_team):
            teams.append(team)

    return teams

# 生成隊伍
teams = generate_team()

# 將隊伍轉換成JSON格式
teams_json = json.dumps(teams, ensure_ascii=False, indent=2)

with open('default_1500_team.json', 'w', encoding="utf-8") as json_file:
    json.dump(teams, json_file, ensure_ascii=False)
# 印出JSON格式的隊伍
print("finish")
