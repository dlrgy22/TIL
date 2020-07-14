import sys
def check_Palindrome(string):
    large_ind = len(string) -1
    small_ind = 0
    while small_ind < large_ind:
        if string[small_ind] != string[large_ind]:
            return False
        small_ind += 1
        large_ind -= 1
    return True

string = sys.stdin.readline()
string = string[:len(string) - 1]
check_string = string
for i in range(len(string)):
    if check_Palindrome(check_string):
        break
    check_string = string
    for j in range(i, -1, -1):
        check_string += string[j]
print(len(check_string))