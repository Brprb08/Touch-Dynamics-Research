from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn import tree 
import pandas as pd
import numpy as np

class preprocess:
    def __init__(self):
        self.X_train: np.array = np.array([])
        self.X_test: np.array = np.array([])
        self.y_train: np.array = np.array([])
        self.y_test: np.array = np.array([])
    
    def __createDf(self, path_incomplete: str, user: int) -> pd.DataFrame:
        df = pd.read_csv(path_incomplete.replace("*", str(user)))

        df = df.loc[ : ,["Timestamp", "X", "Y", "BTN_TOUCH", "FINGER"]]
        df.loc[df["BTN_TOUCH"]=="DOWN", "BTN_TOUCH"] = 0
        df.loc[df["BTN_TOUCH"]=="HELD", "BTN_TOUCH"] = 1
        df.loc[df["BTN_TOUCH"]=="UP", "BTN_TOUCH"] = 2
        df.loc[df["BTN_TOUCH"].isnull(), "BTN_TOUCH"] = 3

        #DROP ALL INDEXES WITH NULL VALUES IN BTN_TOUCH
        df.drop(df.loc[ df["BTN_TOUCH"] == 3, : ].index.astype("int"))
        df["User"] = user

        return df
    
    def tree(self, show: bool = False) -> tree.DecisionTreeClassifier:
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(self.X_train)
        X_test = min_max_scaler.fit_transform(self.X_test)
        
        Y_train = Y_train.astype('int')
        X_train = X_train.astype('int')

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        
        plt.figure(figsize = (12, 12))
        tree.plot_tree(clf, fontsize = 10)
        
        if show:    
            plt.show()

    def getData(self, save: bool = False, path: str = "", save_path: str = "./data.csv"):
        dfs = [self.__createDf(path_incomplete=path, user=(i+1)) for i in range(15)]
        df_all = pd.concat(dfs)

        X = df_all.iloc[:,0:5].values
        Y = df_all.iloc[:, 5].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size = 0.2)
        
        if save:
            df_all.to_csv(save_path)
        
        return X, Y
    
    def openData(self, path: str):
        df = pd.read_csv(path)
        
        X = df.iloc[:,0:5].values
        Y = df.iloc[:, 5].values
        return X, Y
    
if __name__ == "__main__":
    process = preprocess()
    process.getData(save=True, path="pubg_raw/pubg*_touch.csv", save_path="./pubg_all_touch.csv")
