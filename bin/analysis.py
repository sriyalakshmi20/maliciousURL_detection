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

Logmatrix = getAnalysis(143 , 3327 , 426 , 5)
SVMmatrix = getAnalysis(357 , 3324 , 212 , 8)
RandomForest = getAnalysis(338 , 3324 , 231 , 8)
MLP = getAnalysis(446 , 3296 , 123 , 36)
print(Logmatrix)
print(SVMmatrix)
print(RandomForest)
print(MLP)