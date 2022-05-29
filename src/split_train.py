import csv

with open("./data/mnist_train.csv") as f:
    f_scv = csv.reader(f)
    epoch = 0
    line = 0
    while epoch < 6:
        with open("./data/train{}.csv".format(epoch),'w') as ep:
            line = 0
            for row in f_scv:
                line += 1
                w = csv.writer(ep)
                w.writerow(row)
                if line == 10000:
                    break
            epoch += 1