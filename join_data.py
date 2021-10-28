import pandas as pd
import os
path_to_csv_folder = "data/USD_CHF"
# this is the extension you want to detect
extension = '.csv'
#Dataframe to hold all years data
df_2005_2020 = pd.DataFrame()
for root, dirs_list, files_list in os.walk(path_to_csv_folder):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
#             print (file_name)
            print ("Concating",file_name_path)   # This is the full path of the filter file
            df= pd.read_csv(os.path.join(file_name_path),header=None)
            df.columns =["Date","Time","Open","High","Low","Close","Volume"]
            df['Date'] = df['Date'] + " " + df['Time']
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index("Date",inplace=True)
            df_2005_2020 = pd.concat([df_2005_2020,df])

df_2005_2020.to_csv("data/USD_CHF/df_2005_2020.csv")