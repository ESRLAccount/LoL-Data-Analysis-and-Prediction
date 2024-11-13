
import pandas as pd

inputMatchEvent_path1 = "../data/OutputRank/MatchTimeline/"

inputMatchEvent_path2 = "../data/OutputRank/MatchResume/"


inputMatchEvent_path3 = "../data/OutputRank/Spatio-temporal Analysis/"

inputMatchEvent_path4 = "../data/OutputRank/MatchEvents/"

filename=inputMatchEvent_path1 + "MatchTimeline_masterfile_PositionRowwithRoleRankPhase2024.csv"

header = pd.read_csv(filename, nrows=0).columns.tolist()
print(header)



#df = pd.read_csv(filename, nrows=10000)
#print (df)