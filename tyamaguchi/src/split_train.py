import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

pd.options.display.precision = 15

DataPath = Path('input')
DestinationPath = DataPath/'train_split'
DestinationPath.mkdir(exist_ok=True)
header = ['acoustic_data','time_to_failure']
MaxRow = 629145480
rows = 150000


def main():
    train = pd.read_csv(DataPath/'train'/'train.csv')
    segments = int(np.ceil(MaxRow / rows))
    eq_num = 0
    tail = np.inf
    for segment_num in tqdm(range(segments)):
        thisSegment = train[rows*segment_num:rows*(segment_num+1)].copy()
        head = thisSegment['time_to_failure'][rows*segment_num]
        if tail < head:
            eq_num += 1
            thisSegment['eq_num'] = eq_num
        else:
            if segment_num < segments-1:
                min_ind = thisSegment['time_to_failure'].idxmin()
                if min_ind != rows*(segment_num+1)-1:
                    thisSegment['eq_num'] = [eq_num]*(min_ind-rows*segment_num+1)+[eq_num+1]*(rows-(min_ind-rows*segment_num+1))
                    eq_num += 1
                else:
                    thisSegment['eq_num'] = eq_num
                tail = thisSegment['time_to_failure'][rows*(segment_num+1)-1]
            else:
                thisSegment['eq_num'] = eq_num

        thisSegment.to_csv(DestinationPath/'train_{}.csv'.format(segment_num),index=False)

if __name__ == '__main__':
    main()
