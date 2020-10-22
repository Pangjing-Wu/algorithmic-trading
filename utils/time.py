def stamp2time(stamp:int or str) -> str:
    stamp = int(stamp)
    hh = stamp // 3600000
    stamp %= 3600000
    mm = stamp // 60000
    stamp %= 60000
    ss = stamp // 1000
    ms = stamp %  1000
    time = '%02d:%02d:%02d' % (hh, mm, ss)
    time += ':%03d' % ms if ms else ''
    return time

def time2stame(time:str) -> int:
    time = time.split(':')
    if len(time) == 3:
        hh, mm, ss = time
        ms = 0
    elif len(time) == 4:
        hh, mm, ss, ms = time
    else:
        raise KeyError('unknown time format.')
    stamp = 3600000 * int(hh) + 60000 * int(mm) + 1000 * int(ss) + int(ms)
    return stamp