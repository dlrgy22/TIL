def BFS(start, visit, path, tickets):
    for idx in range(len(tickets)):
        if tickets[idx][0] == start and not visit[idx]:
            visit[idx] = True
            path.append(tickets[idx][1])
            BFS(tickets[idx][1], visit, path, tickets)
            if len(path) == len(tickets) + 1:
                return path
            visit[idx] = False
            path.pop()

def solution(tickets):
    tickets = sorted(tickets, key = lambda x : x[1])
    visit = [False for i in range(len(tickets))]
    start = 'ICN'
    path = ['ICN']
    answer = BFS(start, visit, path, tickets)
    return answer

tickets = [['ICN', 'A'], ['A', 'C'], ['A', 'D'], ['D', 'B'], ['B', 'A']]
print(solution(tickets))