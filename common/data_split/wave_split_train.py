from pathlib import Path

DataPath = Path('../../input')
DestinationPath = DataPath / 'train_wave_split'
DestinationPath.mkdir(exist_ok=True)
rows = 150000
delimiter = ','


def main():
    time_ref = 0
    writer = None
    with open(DataPath / 'train' / 'train.csv', 'r') as reader:
        header = next(reader)
        n = 0
        eq_num = 0
        for line in reader:
            if n % (100 * rows) == 0:
                print(n)
            thisData = line.strip().split(delimiter)
            time = float(thisData[1])
            if time_ref < time:
                if time_ref != 0:
                    eq_num += 1
                    writer.close()
                writer = open(
                    DestinationPath / 'train_wave_{}.csv'.format(eq_num), 'w')
                writer.write(header)
            time_ref = time
            writer.write(line)
            n += 1


if __name__ == '__main__':
    main()
