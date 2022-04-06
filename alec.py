import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import  BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import util
import math


def fillMissingValues(data, columnList):

    for column in columnList:
        replacement = -1

        singleValue = data[pd.notnull(data[column])][column].any()
        if type(singleValue) is str:
            replacement = "No Answer"

        data[column].fillna(replacement, inplace=True)

def addVoteColumn(data):
    data["isUnknown"] = ((pd.isna(data["cps19_votechoice"])) & (pd.isna(data["cps19_votechoice_pr"]))
                         & (pd.isna(data["cps19_vote_unlikely"])) & (pd.isna(data["cps19_vote_unlike_pr"]))
                         & (pd.isna(data["cps19_v_advance"])) & (pd.isna(data["cps19_votechoice_7_TEXT"]))
                         & (pd.isna(data["cps19_vote_unlikely_7_TEXT"])) & (
                             pd.isna(data["cps19_vote_unlike_pr_7_TEXT"]))
                         & (pd.isna(data["cps19_v_advance_7_TEXT"])) & (pd.isna(data["cps19_vote_lean"]))
                         & (pd.isna(data["cps19_vote_lean_7_TEXT"])) & (pd.isna(data["cps19_vote_lean_pr"]))
                         & (pd.isna(data["cps19_vote_lean_pr_7_TEXT"])) & (pd.isna(data["cps19_2nd_choice"]))
                         & (pd.isna(data["cps19_2nd_choice_7_TEXT"])) & (pd.isna(data["cps19_2nd_choice_pr"]))
                         & (pd.isna(data["cps19_2nd_choice_pr_7_TEXT"])) & (pd.isna(data["cps19_not_vote_for_1"]))
                         & (pd.isna(data["cps19_not_vote_for_2"])) & (pd.isna(data["cps19_not_vote_for_3"]))
                         & (pd.isna(data["cps19_not_vote_for_4"])) & (pd.isna(data["cps19_not_vote_for_5"]))
                         & (pd.isna(data["cps19_not_vote_for_6"])) & (pd.isna(data["cps19_not_vote_for_7"]))
                         & (pd.isna(data["cps19_not_vote_for_8"])) & (pd.isna(data["cps19_not_vote_for_9"])))

    vote = np.where(pd.isna(data["cps19_votechoice"]), data["cps19_votechoice_pr"], data["cps19_votechoice"])
    vote = np.where(pd.isna(vote), data["cps19_vote_unlikely"], vote)
    vote = np.where(pd.isna(vote), data["cps19_vote_unlike_pr"], vote)
    vote = np.where(pd.isna(vote), data["cps19_v_advance"], vote)
    vote = np.where(pd.isna(vote), data["cps19_vote_lean"], vote)
    vote = np.where(pd.isna(vote), data["cps19_vote_lean_pr"], vote)
    vote = np.where(vote != "I do not intend to vote", vote, None)
    vote = np.where(vote == "ndp", "NPD", vote)
    vote = np.where(vote == "Another party (please specify)", "Autre", vote)
    vote = np.where(vote == "Bloc Qu<e9>b<e9>cois", "BQ", vote)
    vote = np.where(vote == "Don't know/ Prefer not to answer", "NA", vote)
    vote = np.where(vote == "Green Party", "PV", vote)
    vote = np.where(vote == "Liberal Party", "PL", vote)
    vote = np.where(vote == "Conservative Party", "PC", vote)
    vote = np.where(vote == "People's Party", "PP", vote)
    data["vote"] = vote

    data.drop(columns=["cps19_votechoice", "cps19_votechoice_pr", "cps19_vote_unlikely", "cps19_vote_unlike_pr",
                       "cps19_v_advance", "cps19_votechoice_7_TEXT", "cps19_vote_unlikely_7_TEXT",
                       "cps19_vote_unlike_pr_7_TEXT", "cps19_v_advance_7_TEXT", "cps19_vote_lean",
                       "cps19_vote_lean_7_TEXT", "cps19_vote_lean_pr", "cps19_vote_lean_pr_7_TEXT", "cps19_2nd_choice",
                       "cps19_2nd_choice_7_TEXT", "cps19_2nd_choice_pr", "cps19_2nd_choice_pr_7_TEXT",
                       "cps19_not_vote_for_1", "cps19_not_vote_for_2", "cps19_not_vote_for_3", "cps19_not_vote_for_4",
                       "cps19_not_vote_for_5", "cps19_not_vote_for_6", "cps19_not_vote_for_7", "cps19_not_vote_for_8",
                       "cps19_not_vote_for_9", "cps19_votechoice_pr_7_TEXT", "cps19_not_vote_for_7_TEXT"], inplace=True)
    return data


def dateToInt(data, column):
    X = data[column].to_numpy()
    Y = np.array([datetime.strptime(Xi, '%Y-%m-%d %H:%M:%S').timestamp() for Xi in X])
    data[column] = Y
    return data


def stringToOneHot(X, X_unknown, columns):
    for column in columns:
        onehot = pd.get_dummies(X[column], prefix=column)
        onehotUnknown = pd.get_dummies(X_unknown[column], prefix=column)
        for oneHotColumn in onehot:
            X[oneHotColumn] = onehot[oneHotColumn]
            if oneHotColumn in onehotUnknown.columns:
                X_unknown[oneHotColumn] = onehotUnknown[oneHotColumn]
            else:
                X_unknown[oneHotColumn] = 0
        X.drop(columns=[column], inplace=True)
        X_unknown.drop(columns=[column], inplace=True)

def clean_boolean(data, columns):
    data_is_present = pd.notnull(data[columns[0]])
    for column in columns:
        data_is_present = data_is_present | (pd.notnull(data[column]))

    for column in columns:
        data[column] = np.where(data_is_present & pd.notnull(data[column]), True, data[column])
        data[column] = np.where(data_is_present & pd.isna(data[column]), False, data[column])

    return data

def findSucessWithoutNaN(data, columnToUse, answeredGiven):
    le = LabelEncoder()

    answersToUse  = data[columnToUse][~data[columnToUse].isna()][~data['vote'].isna()]
    vote = data["vote"][~data[columnToUse].isna()][~data['vote'].isna()]

    nbrValidAnswers = len(answersToUse)

    voteEncoded = le.fit_transform(vote)
    le_vote_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    #print(le_vote_mapping)

    answersEncoded = le.fit_transform(answersToUse)
    le_answer_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    #print(le_answer_mapping)

    X_train, X_test, y_train, y_test = train_test_split(answersEncoded, voteEncoded, test_size=0.20, random_state=1)

    clf = GaussianNB()
    clf.fit(X_train.reshape(-1, 1), y_train)
    y_pred = clf.predict(X_test.reshape(-1, 1))
    successRate = accuracy_score(y_true=y_test, y_pred = y_pred)

    testPrediction = le_answer_mapping[answeredGiven]

    predictedAnswerNumber = clf.predict(np.array([testPrediction]).reshape(-1, 1))
    predictedAnswer = ""
    for vote, voteMappping in le_vote_mapping.items():
        if predictedAnswerNumber == voteMappping:
            predictedAnswer = vote
    print("Reponse prevue pour: {question}: {answer} avec une fiabilite de {accuracy} et {nbrRepondants} repondants.".format(
        question=columnToUse, answer = predictedAnswer, accuracy = successRate, nbrRepondants = nbrValidAnswers))

    return predictedAnswer, successRate, nbrValidAnswers


def getFullResults(data, persons):

    #Iteratate over all desired voters
    allVotersResults = {}
    for index, row in persons.iterrows():
        #Iterate over all desired answers
        print("NOUVELLE PERSONNE NUMERO : {index}! ".format(index=index))
        voterResults = []

        for i in range(0, len(row)):
            isNaN = True
            try:
                isNaN = math.isnan(float(row[i]))
            except ValueError as err:
                isNaN = False

            if not isNaN:
                predictedAnswer, successRate, nbrValidAnswers = findSucessWithoutNaN(data, row.index[i], row[i])
                results = (predictedAnswer, successRate, nbrValidAnswers)
                voterResults.append(results)

        allVotersResults[index] = voterResults
    return allVotersResults

def getCombinedResults(allResults):
    combinedResultsDict = {}
    for voter, listOfResults in allResults.items():
        combinedResults = {'Autre':[0, 0], 'BQ': [0, 0], 'NA': [0, 0], 'NPD': [0, 0], 'PC': [0, 0], 'PL': [0, 0], 'PP': [0, 0], 'PV': [0, 0]}
        for resultToQuestion in listOfResults:
            predictedVote = resultToQuestion[0]
            combinedResults[predictedVote][0] +=  resultToQuestion[1]
            combinedResults[predictedVote][1] += resultToQuestion[2]
        combinedResultsDict[voter] = combinedResults
    return combinedResultsDict

def getFinalPredictions(combinedResults):
    finalVotePredictionsDict = {}
    for voter, combinedResultsDict in combinedResults.items():
        #print(voter, '->', combinedResultsDict)

        highestVoteProbability = 0
        finalVotePrediction = "PL"
        for possibleVote, possibleVoteStats in combinedResultsDict.items():
            #print(possibleVote, '->', possibleVoteStats)
            if(possibleVoteStats[0] > highestVoteProbability):
                #Put logic for vote here with some formulas
                highestVoteProbability = possibleVoteStats[0]
                finalVotePrediction = possibleVote

        finalVotePredictionsDict[voter] = [finalVotePrediction, highestVoteProbability]
    return finalVotePredictionsDict

def main():

    data = pd.read_csv("CES19.csv")
    data = addVoteColumn(data)

    DATE_COLUMNS = ["cps19_StartDate"]
    COLUMNS_TO_CLUSTER = ["cps19_StartDate", "cps19_party_rating_23"]
    COLUMNS_STRING_TO_INT = ['cps19_province', 'cps19_outcome_most', 'cps19_imp_iss_party', 'cps19_fed_id',
                             'cps19_fed_member', 'cps19_fed_gov_sat', 'pid_en', 'pid_party_en', 'pid_party_fr',
                             'cps19_prov_id', 'cps19_vote_2015']
    COLUMNS_FOR_CLASSIFICATION = ["cps19_StartDate", 'cps19_province', 'cps19_outcome_most', 'cps19_imp_iss_party', 'cps19_fed_id',
                             'cps19_fed_member', 'cps19_fed_gov_sat', 'pid_en', 'pid_party_en', 'pid_party_fr',
                             'cps19_prov_id', 'cps19_vote_2015', "cps19_party_rating_23", "cps19_party_rating_23", "cps19_party_rating_24", "cps19_party_rating_25", "cps19_party_rating_26", "cps19_party_rating_27", "cps19_party_rating_28",
                             "cps19_lead_rating_23", "cps19_lead_rating_24", "cps19_lead_rating_25", "cps19_lead_rating_26", "cps19_lead_rating_27", "cps19_lead_rating_28"]
    BOOLEANS_TO_CLEAN = [["cps19_lead_int_113", "cps19_lead_int_114", "cps19_lead_int_115", "cps19_lead_int_116",
                          "cps19_lead_int_117", "cps19_lead_int_118", "cps19_lead_int_119", "cps19_lead_int_120"],
                         ["cps19_lead_strong_113", "cps19_lead_strong_114", "cps19_lead_strong_115",
                          "cps19_lead_strong_116", "cps19_lead_strong_117", "cps19_lead_strong_118",
                          "cps19_lead_strong_119", "cps19_lead_strong_120"],
                         ["cps19_lead_trust_113", "cps19_lead_trust_114", "cps19_lead_trust_115",
                          "cps19_lead_trust_116",
                          "cps19_lead_trust_117", "cps19_lead_trust_118", "cps19_lead_trust_119",
                          "cps19_lead_trust_120"],
                         ["cps19_lead_cares_113", "cps19_lead_cares_114", "cps19_lead_cares_115",
                          "cps19_lead_cares_116",
                          "cps19_lead_cares_117", "cps19_lead_cares_118", "cps19_lead_cares_119",
                          "cps19_lead_cares_120"]]
    COLUMNS_TO_FILL = ["cps19_party_rating_23", "cps19_party_rating_24", "cps19_party_rating_25", "cps19_party_rating_26", "cps19_party_rating_27", "cps19_party_rating_28", "cps19_lead_rating_23", "cps19_lead_rating_24", "cps19_lead_rating_25", "cps19_lead_rating_26", "cps19_lead_rating_27", "cps19_lead_rating_28"]

    for column in DATE_COLUMNS:
        data = dateToInt(data, column)

    for group in BOOLEANS_TO_CLEAN:
        data = clean_boolean(data, group)


    fillMissingValues(data, COLUMNS_TO_FILL)

    X = data[pd.notnull(data['vote'])][COLUMNS_FOR_CLASSIFICATION]
    y = data[pd.notnull(data['vote'])]['vote']
    unknown_indices = util.readResultFile()
    X_unknown = data.iloc[unknown_indices][COLUMNS_FOR_CLASSIFICATION]

    stringToOneHot(X, X_unknown, COLUMNS_STRING_TO_INT)

    #This is the code for homemade Baysian classifier after manually selecting questions and removing uninteresting answers.
    onlyDesiredAnswers = data.loc[:, ['cps19_outcome_most', 'cps19_province', 'cps19_imp_iss_party', 'cps19_fed_id', 'cps19_fed_member',
                        'cps19_fed_gov_sat', 'pes19_provvote', 'cps19_prov_member', 'cps19_party_rating_24',
                        'cps19_lead_rating_24', 'cps19_cand_rating_24', 'pid_party_fr', 'pid_en', 'cps19_cand_rating_26',
                        'cps19_party_rating_25', 'pes19_votechoice2019']]

    #Temporary persons used for testing.
    # TODO: Test method with real test samples
    persons = onlyDesiredAnswers.iloc[[1, 3, 5, 7, 9, 11]]
    allResults = getFullResults(data, persons)
    combinedResults = getCombinedResults(allResults)
    finalVotePredictions = getFinalPredictions(combinedResults)
    print(finalVotePredictions)

    """
    #This is the code for usual, general Baysian classifier
    scores = []
    for i in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        successRate = accuracy_score(y_true=y_test, y_pred=y_pred)

        scores.append(successRate)

        predictions = clf.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
            display_labels = clf.classes_)
        disp.plot()
        plt.show()
        util.writeResultFile(X_unknown, clf)
    
    avgScore = np.array(scores).mean()
    print("Score: " + str(avgScore))
    """

if __name__ == '__main__':
    main()