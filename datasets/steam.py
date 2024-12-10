import pandas as pd


with open('Steam.txt', 'r') as f:
    lines = f.readlines()
    user_list = []
    item_list = []
    for line in lines:
        list_temp = line.strip('\n').split(' ')
        user_list.append(list_temp[0])
        item_list.append(list_temp[1])

steam_csv = pd.DataFrame({'uid': user_list, 'sid': item_list})


steam_csv.to_csv('ratings_Steam.csv', index=False, header=False)
