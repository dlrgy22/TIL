import sys

def check_n():
    if n == 1:
        print(0)
        sys.exit()
    elif n == 2:
        print(1)
        sys.exit()

def find_prime():
    prime_list = []
    prime_check = [True for i in range(n + 1)]
    for i in range(2, n + 1):
        if prime_check[i]:
            prime_list.append(i)
            idx = 2
            while i * idx <= n:
                prime_check[i * idx] = False
                idx += 1
    return prime_list

def sum_prime(prime_list, sum, idx):
    global count
    while sum > n:
        sum -= prime_list[idx]
        idx += 1
    if sum == n:
        count += 1
    return sum, idx

n = int(sys.stdin.readline())
check_n()
prime_list = find_prime()
count = 0
start_idx = 0
sum = 0
idx = 0

while sum + prime_list[idx] <= n:
    sum += prime_list[idx]
    idx += 1
if sum == n:
    count += 1
end_idx = idx

while end_idx < len(prime_list):
    sum, start_idx = sum_prime(prime_list, sum + prime_list[end_idx], start_idx)
    end_idx += 1

print(count)