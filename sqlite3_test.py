import sys
import pandas as pd
import sqlite3

sys.stderr.write("*** 開始 ***\n")

df = pd.DataFrame([
        ["t1271","千葉",51476,"2003-9-25"],
        ["t1272","勝浦",42573,"2003-3-16"],],
                  index=['row1', 'row2'],
                  columns=['a', 'b', 'c', 'd'])

print(df)
file_sqlite3 = "./cities.db"
conn = sqlite3.connect(file_sqlite3)
df.to_sql('cities',conn,if_exists='append',index=None)
conn.close()
sys.stderr.write("*** 終了 ***\n")
