import sys
import queue
def Manhatan_distance(location1,location2):
    return abs(location1[0]-location2[0]) + abs(location1[1] - location2[1])
def BFS(start,conv,end):
    visit = [False for i in range(n)]
    q = queue.Queue()
    q.put(start)
    while q.qsize() != 0:
        location = q.get()
        if Manhatan_distance(location, end) <= 1000:
            print("happy")
            return
        else:
            for i in range(n):
                if visit[i]:
                    continue
                elif Manhatan_distance(location,conv[i]) <= 1000:
                    q.put(conv[i])
                    visit[i] = True
    print("sad")
t = int(sys.stdin.readline())
for i in range(t):
    n = int(sys.stdin.readline())
    home_x,home_y = map(int,sys.stdin.readline().split())
    conv = []
    for i in range(n):
        add_x,add_y = map(int,sys.stdin.readline().split())
        conv.append([add_x,add_y])
    fest_x,fest_y = map(int,sys.stdin.readline().split())
    BFS([home_x,home_y],conv,[fest_x,fest_y])