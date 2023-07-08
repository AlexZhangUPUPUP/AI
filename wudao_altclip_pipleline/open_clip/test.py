ilist = list(range(11))

def iterd(batch):
    count = []
    for i in batch:
        count.append(i)
        if len(count) == 3:
            yield count
            count = []
    yield count

for ir in  iterd(ilist):
    print(ir)
