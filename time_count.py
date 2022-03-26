import time
global time_start

t=0
time_start=time.time()

def time_report():
    global t
    global time_start
    if (t == 0):
        time_start=time.time()
        t=t+1
    else:
        time_end = time.time()
        print('Done,it cost :',int(time_end-time_start)," seconds")
        time_start=time.time()



