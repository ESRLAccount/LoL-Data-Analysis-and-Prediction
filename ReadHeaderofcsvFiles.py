
import pandas as pd

inputMatchEvent_path1 = "../data/OutputRank/MatchTimeline/"

inputMatchEvent_path2 = "../data/OutputRank/MatchResume/"


inputMatchEvent_path3 = "../data/OutputRank/Spatio-temporal Analysis/"

inputMatchEvent_path4 = "../data/OutputRank/MatchEvents/"

filename=inputMatchEvent_path2 + "RankedMatchResume_Masterfile2024.csv"

header = pd.read_csv(filename, nrows=0).columns.tolist()
print(header)


