file = 'data/stocklist/SH50.txt'

with open(file, 'r') as f:
    string = f.read()
    nums = [str(n) for n in range(10)]
    ret = ''
    i = 0
    for s in string:
        if s in nums:
            if i % 6 == 0 and i != 0:
                ret += '\n'
            ret += s
            i += 1
    print(ret)