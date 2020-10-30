from collections import deque

def solution(n, computers):
    q = deque()
    answer = 0
    visit = [False for i in range(n)]

    for i in range(n):
        q = deque()
        if not visit[i]:
            q.append(i)
            while len(q) != 0:
                node = q.popleft()
                for idx in range(n):
                    if computers[node][idx] == 1 and not visit[idx]:
                        q.append(idx)
                        visit[idx] = True
            answer += 1

    return answer

n = 3
computers = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
print(solution(n, computers))
