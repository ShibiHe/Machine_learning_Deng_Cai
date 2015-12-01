w = sorted([19957, 56930, 13613, 37568, 65398, 9494, 45153, 38176, 75526, 30033])
f = open('all_word_map.txt', 'r')
i = 0
for line in f:
    a = line.strip().split()
    if w[i] == int(a[1]):
        print(a[0], end=' ')
        i = i + 1
    if i > 9:
        break
