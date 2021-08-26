from datetime import datetime as dt

flag = False

while(1):

    time_now = dt.now()
    if not time_now.second % 5 and flag == True:
        print(time_now)
        flag = False
    elif not time_now.second % 5 and flag == False:
        pass
    else:
        flag = True




