from queue import PriorityQueue

def solution(jobs):
    answer = 0
    heap = PriorityQueue()
    jobs = sorted(jobs)
    time = jobs[0][0]
    i = 0
    while i != len(jobs):
        if jobs[i][0] <= time:
            heap.put((jobs[i][1], jobs[i][0]))
            i += 1
        else:
            if heap.qsize() != 0:
                spend_time, input_time = heap.get()
                time += spend_time
                answer += time - input_time
                if time > jobs[i][0]:
                    heap.put((jobs[i][1], jobs[i][0]))
                    i += 1
            else:
                time = jobs[i][0]
                heap.put((jobs[i][1], jobs[i][0]))
                i += 1

    while heap.qsize() != 0:
        spend_time, input_time = heap.get()
        time += spend_time
        answer += (time - input_time)


    return answer // len(jobs)

print(solution([[0, 10], [100, 10], [200, 10]]))
