import sys
import pandas as pd
import sqlite3, pdb

sys.stderr.write("*** 開始 ***\n")

df = pd.DataFrame([
        ["t1271","千葉",51476,"2003-9-25"],
        ["t1272","勝浦",42573,"2003-3-16"],],
                  index=['row1', 'row2'],
                  columns=['a', 'b', 'c', 'd'])

file_sqlite3 = "./cities.db"

conn = sqlite3.connect(file_sqlite3)
df.to_sql('cities', conn, if_exists='append', index=None)
conn.close()

df.loc['row3'] = {'a': 't1273', 'b': '西宮', 'c': '23423', 'd': '2021-08-02'}

conn = sqlite3.connect(file_sqlite3)
df.tail(1).to_sql('cities', conn, if_exists='append', index=None)
conn.close()

sys.stderr.write("*** 終了 ***\n")
