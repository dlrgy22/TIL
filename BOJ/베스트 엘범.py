def solution(genres, plays):
    answer = []
    genres_dic = {}
    for idx in range(len(plays)):
        if genres[idx] in genres_dic:
            total_play, large_1, large_2 = genres_dic[genres[idx]]
            total_play += plays[idx]
            if plays[idx] > large_1[1]:
                large_2 = [large_1[0], large_1[1]]
                large_1 = [idx, plays[idx]]
            elif plays[idx] > large_2[1]:
                large_2 = [idx, plays[idx]]

            genres_dic[genres[idx]] = [total_play, large_1, large_2]
        else:
            genres_dic[genres[idx]] = [plays[idx], [idx, plays[idx]], [-1, -1]]

    best_album = list(genres_dic.items())
    best_album = sorted(best_album, key = lambda x : x[1][0], reverse = True)

    for genre, data in best_album:
        answer.append(data[1][0])
        if data[2][0] != -1:
            answer.append(data[2][0])
    return answer

genres = ['a', 'a', 'a', 'd']
plays = [2000, 500, 500, 2500]
print(solution(genres, plays))