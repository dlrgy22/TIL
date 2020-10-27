from collections import deque
def cycle_check(n1, n2, visit, n):
    result = False
    q = deque()
    q.append(n1)
    visit_node = [False for i in range(n)]
    visit_node[n1] = True
    while len(q) != 0:
        node = q.popleft()
        if node == n2:
            result = True
            break
        for i in range(n):
            if visit[node][i] and visit_node[i] == False:
                q.append(i)
                visit_node[i] = True
    return result



def solution(n, costs):
    answer = 0
    count = 0
    costs = sorted(costs, key = lambda x : x[2])
    visit = [[False for j in range(n)] for i in range(n)]
    for cost in costs:
        if not cycle_check(cost[0], cost[1], visit, n):
            visit[cost[0]][cost[1]] = True
            visit[cost[1]][cost[0]] = True
            count += 1
            answer += cost[2]
            if count == n - 1:
                break

    return answer

costs = [[0,1,1],[0,2,2],[1,2,5],[1,3,1],[2,3,8]]
print(solution(4, costs))

