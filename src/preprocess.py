import pandas as pd
import numpy as np
import os
import glob
from collections import deque
import random
from pathlib import Path


def main():
    path = '/Users/adminuser/Desktop/Research Code/Touch-Dynamics-Research/pubg_raw'
    filelist = [file for file in os.listdir(path) if file.endswith('.csv')]
    for file in filelist:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        df = pd.read_csv(path + '/' + file)
        df.drop('BTN_TOUCH', inplace=True, axis=1)
        df.insert(len(df.columns), "X_Speed", 0)
        df.insert(len(df.columns), "X_Acceleration", 0)
        df.insert(len(df.columns), "Y_Speed", 0)
        df.insert(len(df.columns), "Y_Acceleration", 0)
        df.insert(len(df.columns), "Speed", 0)
        df.insert(len(df.columns), "Acceleration", 0)
        df.insert(len(df.columns), "Jerk", 0)
        df.insert(len(df.columns), "Ang_V", 0)
        # print(df.head)
        #df.insert(len(df.columns), "EMPTY", 0)
        df['X_Speed'] = (df.X - df.X.shift(1)) / \
            (df.Timestamp - df.Timestamp.shift(1))
        df['Y_Speed'] = (df.Y - df.Y.shift(1)) / \
            (df.Timestamp - df.Timestamp.shift(1))
        df['Speed'] = np.sqrt((df.X_Speed ** 2) + (df.Y_Speed ** 2))
        df['X_Acceleration'] = (
            df.X_Speed - df.X_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
        df['Y_Acceleration'] = (
            df.Y_Speed - df.Y_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
        df['Acceleration'] = (df.Speed - df.Speed.shift(1)) / \
            (df.Timestamp - df.Timestamp.shift(1))
        df['Jerk'] = (df.Acceleration - df.Acceleration.shift(1)
                      ) / (df.Timestamp - df.Timestamp.shift(1))
        df['Path_Tangent'] = np.arctan2(
            (df.Y - df.Y.shift(1)), (df.X - df.X.shift(1)))
        df['EMPTY'] = 0  # place holder to avoid out of bounds errors?
        df['Ang_V'] = (df.Path_Tangent - df.Path_Tangent.shift(1)
                       ) / (df.Timestamp - df.Timestamp.shift(1))
        df.set_index("Timestamp", inplace=True)
        df = df.drop(labels=0, axis=0)
        df = df.fillna("0")
        #df = df.drop(labels=1, axis=0)
        sequential_data = []
        SEQ_LEN = 10
        prev_data = deque(maxlen=SEQ_LEN)
        for i in df.values:
            # print(df.head)
            usrID = 1  # instead of 1 --> df.iloc[5]['CLASS']
            # Append each row in df to prev_data without 'Subject ID' column, up to 60 rows
            prev_data.append([n for n in i[:-1]])
            if len(prev_data) == SEQ_LEN:
                temp = np.copy(prev_data)
                temp = temp.astype(float)
                for j in range(6, 14):
                    temp[0, j] = 0

                mean_touch_major = temp[1:, 2].mean()
                std_touch_major = temp[1:, 2].std()
                min_touch_major = temp[1:, 2].min()
                max_touch_major = temp[1:, 2].max()

                mean_touch_minor = temp[1:, 3].mean()
                std_touch_minor = temp[1:, 3].std()
                min_touch_minor = temp[1:, 3].min()
                max_touch_minor = temp[1:, 3].max()

                mean_x_speed = temp[1:, 6].mean()
                std_x_speed = temp[1:, 6].std()
                min_x_speed = temp[1:, 6].min()
                max_x_speed = temp[1:, 6].max()

                mean_y_speed = temp[1:, 7].mean()
                std_y_speed = temp[1:, 7].std()
                min_y_speed = temp[1:, 7].min()
                max_y_speed = temp[1:, 7].max()

                mean_speed = temp[1:, 8].mean()
                std_speed = temp[1:, 8].std()
                min_speed = temp[1:, 8].min()
                max_speed = temp[1:, 8].max()

                mean_x_acc = temp[1:, 9].mean()
                std_x_acc = temp[1:, 9].std()
                min_x_acc = temp[1:, 9].min()
                max_x_acc = temp[1:, 9].max()

                mean_y_acc = temp[1:, 10].mean()
                std_y_acc = temp[1:, 10].std()
                min_y_acc = temp[1:, 10].min()
                max_y_acc = temp[1:, 10].max()

                mean_acc = temp[1:, 11].mean()
                std_acc = temp[1:, 11].std()
                min_acc = temp[1:, 11].min()
                max_acc = temp[1:, 11].max()

                mean_jerk = temp[1:, 12].mean()
                std_jerk = temp[1:, 12].std()
                min_jerk = temp[1:, 12].min()
                max_jerk = temp[1:, 12].max()

                mean_tan = temp[1:, 13].mean()
                std_tan = temp[1:, 13].std()
                min_tan = temp[1:, 13].min()
                max_tan = temp[1:, 13].max()

                mean_ang = temp[1:, 14].mean()
                std_ang = temp[1:, 14].std()
                min_ang = temp[1:, 14].min()
                max_ang = temp[1:, 14].max()

                for jj in [[mean_x_speed, mean_y_speed, mean_speed, mean_x_acc, mean_y_acc, mean_acc, mean_jerk, mean_tan, mean_ang, mean_touch_major, mean_touch_minor,
                            std_x_speed, std_y_speed, std_speed, std_x_acc, std_y_acc, std_acc, std_ang, std_tan, std_jerk, std_touch_major, std_touch_minor,
                            min_x_speed, min_y_speed, min_speed, min_x_acc, min_y_acc, min_acc, min_ang, min_tan, min_jerk, min_touch_major, min_touch_minor,
                            max_x_speed, max_y_speed, max_speed, max_x_acc, max_y_acc, max_acc, max_ang, max_tan, max_jerk, max_touch_major, max_touch_minor, usrID]]:  # 44 values
                    # Prev_data now contains SEQ_LEN amount of samples and can be appended as one batch of 60 for RNN
                    sequential_data.append(jj)

        df = pd.DataFrame(sequential_data,
                          columns=['mean_x_speed', 'mean_y_speed', 'mean_speed', 'mean_x_acc', 'mean_y_acc', 'mean_acc', 'mean_jerk', 'mean_tan', 'mean_ang', 'mean_touch_major', 'mean_touch_minor',
                                   'std_x_speed', 'std_y_speed', 'std_speed', 'std_x_acc', 'std_y_acc', 'std_acc', 'std_ang', 'std_tan', 'std_jerk', 'std_touch_major', 'std_touch_minor',
                                   'min_x_speed', 'min_y_speed', 'min_speed', 'min_x_acc', 'min_y_acc', 'min_acc', 'min_ang', 'min_tan', 'min_jerk', 'min_touch_major', 'min_touch_minor',
                                   'max_x_speed', 'max_y_speed', 'max_speed', 'max_x_acc', 'max_y_acc', 'max_acc', 'max_ang', 'max_tan', 'max_jerk', 'max_touch_major', 'max_touch_minor', 'class'])
        df = df.sample(frac=1)
        train_df, test_df = np.split(df, [int(.8*len(df))])

        name_no_ext = os.path.splitext(file)[0]
        print(name_no_ext)
        train_df.to_csv(name_no_ext + "_train.csv", index=False)
        test_df.to_csv(name_no_ext + "_test.csv", index=False)
        random.shuffle(sequential_data)

    #path = 'C:/Users/zachd/Documents/ATA-multi touch/ZModifiedData'
    #filelist = [file for file in os.listdir(path) if file.endswith('.csv')]
    # for file in filelist:
    #        df = pd.read_csv(path + '/' + file)
    #        df.drop('ORIENTATION', inplace=True, axis=1)
    #        df.drop('PRESSURE', inplace=True, axis=1)
    #        df['BTN_TOUCH'] = df['BTN_TOUCH'].fillna("HELD")
    #        df.to_csv(file, index=False)


if __name__ == '__main__':
    main()

<<<<<<< HEAD:preprocess.py
# The TouchMajor and TouchMinor fields describe the approximate dimensions of the contact area in output units (pixels).
=======
#The TouchMajor and TouchMinor fields describe the approximate dimensions of the contact area in output units (pixels).
>>>>>>> c4b50f8d3c99891baf8cb092d73fa303c7797a51:src/preprocess.py
