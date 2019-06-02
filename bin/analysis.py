'''
We are calculating the TPR, FPR, FNR, and TNR

'''

def getAnalysis(TP,TN,FP,FN):
    analysisMatrix = []
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    FPR = FP/(FP + TN)
    FNR = FN/(FN + TP)
    analysisMatrix.append(TPR)
    analysisMatrix.append(TNR)
    analysisMatrix.append(FPR)
    analysisMatrix.append(FNR)
    return analysisMatrix

Logmatrix = getAnalysis(9980,1430,44,24)
SVMmatrix = getAnalysis(9912,1453,22,1)
RandomForest = getAnalysis(9908,1454,26,0)
MLP = getAnalysis(9915,1454,19,0)
print(Logmatrix)
print(SVMmatrix)
print(RandomForest)
print(MLP)