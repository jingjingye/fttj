# coding=utf-8

if __name__ == "__main__":
    #################simCase####################
    # from flow.similarCases import trainSimilarCases
    # trainSimilarCases(True, "tfidf")
    # for i in range(10, 310, 10):
    #     trainSimilarCases(False, "lda", i)
    # trainSimilarCases(False, "svd", 70)

    #################vector####################
    # from flow.wordvector import trainWordvector
    # trainWordvector()

    #################cnn####################
    # from flow.cnn import trainDataPrepare, trainCnn
    # trainDataPrepare("lda")
    # trainCnn("ann")

    ##################lr####################
    from flow.lr import trainDataPrepare, trainLR
    trainDataPrepare("ann", 35, "lda")
    # trainLR()

    #################rules####################
    # from flow.rules import trainRules
    # trainRules(1, 0.7)

    #################constract################
    # from flow.multiLabel import prepareLabels, trainMulti
    # prepareLabels(4)
    # trainMulti("svm")

    #################f1#######################
    # a = float(input())
    # b = float(input())
    # c = 2*a*b/(a+b)
    # print(str(c))
