import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import graphviz
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import tree
import os
import util
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
import random
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

print(pd.__version__)

desired_width = 1080
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 60)

VOTE_QUESTIONS = ["cps19_votechoice", "cps19_votechoice_pr", "cps19_vote_unlikely", "cps19_vote_unlike_pr",
                  "cps19_v_advance", "cps19_votechoice_7_TEXT", "cps19_vote_unlikely_7_TEXT",
                  "cps19_vote_unlike_pr_7_TEXT",
                  "cps19_v_advance_7_TEXT", "cps19_vote_lean", "cps19_vote_lean_7_TEXT", "cps19_vote_lean_pr",
                  "cps19_vote_lean_pr_7_TEXT", "cps19_2nd_choice", "cps19_2nd_choice_7_TEXT", "cps19_2nd_choice_pr",
                  "cps19_2nd_choice_pr_7_TEXT", "cps19_not_vote_for_1", "cps19_not_vote_for_2", "cps19_not_vote_for_3",
                  "cps19_not_vote_for_4", "cps19_not_vote_for_5", "cps19_not_vote_for_6", "cps19_not_vote_for_7",
                  "cps19_not_vote_for_8", "cps19_not_vote_for_9"]


def questionnaire(data):
    columns = data.columns

    completeCount = 0
    missingValues = []

    for column in columns:
        missingValuesCount = 0
        for datapoint in data[column]:
            if pd.isna(datapoint):
                missingValuesCount += 1
        if missingValuesCount == 0:
            completeCount = completeCount + 1
        missingValues.append(missingValuesCount)

    print(completeCount)

    plt.hist(missingValues)
    plt.show()

    datesDuSondage = data['cps19_StartDate']
    dateElection = pd.Timestamp(2019, 10, 21)
    dateLendemain = pd.Timestamp(2019, 10, 22)

    lateCount = 0
    for d in datesDuSondage:

        if pd.Timestamp(d).date() == dateElection.date() or pd.Timestamp(d).date() == dateLendemain.date():
            lateCount += 1

    print(lateCount)

    Q7 = len(data[(data["cps19_yob"] == 1979) | (data["cps19_citizenship"] == "Permanent resident")]
             [["cps19_citizenship", "cps19_yob"]])
    print(Q7)

    Q8 = data.iloc[101][["cps19_citizenship", "cps19_yob"]]
    print(Q8)

    optimists = 0
    dataInt = data[(pd.isna(data["cps19_lead_int_119"])) & (pd.isna(data["cps19_lead_int_120"]))][["cps19_lead_int_113",
                                                                                                   "cps19_lead_int_114",
                                                                                                   "cps19_lead_int_115",
                                                                                                   "cps19_lead_int_116",
                                                                                                   "cps19_lead_int_117",
                                                                                                   "cps19_lead_int_118"]]
    for index, row in dataInt.iterrows():
        count = 0
        for leader in row:
            if not pd.isna(leader):
                count += 1
        if count >= 3:
            optimists += 1
    print(optimists)

    rhinos = 0
    dataOther = data[data["cps19_votechoice"] == "Another party (please specify)"]["cps19_votechoice_7_TEXT"]
    for text in dataOther:
        if "rhino" in text or "Rhino" in text:
            print("------------------------------" + text)
            rhinos += 1
        else:
            print(text)
    print(rhinos)


def chisquare(data):
    total = data.sum().sum()
    chi2 = 0

    for column in data.columns:
        column_sum = data[column].sum()
        for index in data.index:
            index_sum = data.loc[index].sum()
            expected = (column_sum * index_sum) / total
            actual = data.loc[index][column]
            chi2 += np.power((actual - expected), 2) / expected

    return chi2


def score_contingency(contingency, data, column_name):
    test_data = data[data["isUnknown"] == False]
    total_responses = test_data["Unnamed: 0"].count()
    top_vote = test_data["vote"].value_counts().idxmax()
    top_vote_score = test_data["vote"].value_counts().max() / total_responses

    answerTypes = len(contingency.columns)
    if answerTypes >= 20:
        return top_vote_score

    good_answers = 0
    for column in contingency:
        good_answers += contingency[column].max()

    good_answers += test_data[pd.isna(test_data[column_name]) & (test_data["vote"] == top_vote)]["vote"].count()
    score = good_answers / total_responses
    return score


def get_best_questions(data, getMin=False):
    test_data = data[data["isUnknown"] == False]
    total_responses = test_data["Unnamed: 0"].count()
    min_responses = total_responses / 1000
    results = pd.DataFrame([], columns=['column', 'answer', 'vote', 'score', 'size'])

    for colonne in data:
        if colonne in VOTE_QUESTIONS:
            continue
        data_colonne = data[colonne]

        try:

            if data_colonne.dtype == 'int64':
                data_colonne = pd.qcut(data_colonne, 10, labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

            #            answerTypes = len(data_colonne.value_counts())
            #            if answerTypes >= 20:
            #                continue

            contingency = pd.crosstab(data['vote'], data_colonne)

            for answer in contingency:
                voters = contingency[answer].max()
                if voters < min_responses:
                    continue
                vote = contingency[answer].idxmax()
                total = contingency[answer].sum()
                score = voters / total
                if vote not in ["Conservative Party", "Liberal Party"]:
                    results = results.append(pd.DataFrame([[colonne, answer, vote, score, total]],
                                                          columns=['column', 'answer', 'vote', 'score', 'size']))

                if getMin:
                    votersmin = contingency[answer].min()
                    votemin = contingency[answer].idxmin()
                    scoremin = votersmin / total
                    results = results.append(pd.DataFrame([[colonne, answer, votemin, scoremin, total]],
                                                          columns=['column', 'answer', 'vote', 'score', 'size']))

        except:
            print(f"Colonne {colonne} fail")

    results = results.sort_values(by=['score'], ascending=False)
    for column, answer, vote, score, size in results.values:
        print(f"{column} & {answer} & {vote} & {score:.2f} & {size:.0f}")


def clean_boolean(data, columns):
    data_is_present = pd.notnull(data[columns[0]])
    for column in columns:
        data_is_present = data_is_present | (pd.notnull(data[column]))

    for column in columns:
        data[column] = np.where(data_is_present & pd.notnull(data[column]), True, data[column])
        data[column] = np.where(data_is_present & pd.isna(data[column]), False, data[column])

    return data


def exploration(data, verbose=False):
    results = pd.DataFrame([], columns=['Column', 'p', 'score'])

    for colonne in data:
        print("Colonne: " + colonne)
        data_colonne = data[colonne]
        if verbose:
            print(data_colonne.value_counts())

        try:

            if data_colonne.dtype == 'int64':
                data_colonne = pd.qcut(data_colonne, 10)

            contingency = pd.crosstab(data['vote'], data_colonne)

            c, p, dof, expected = stats.chi2_contingency(contingency)

            if p < 0.05:
                if verbose:
                    contingency_percent = pd.crosstab(data['vote'], data_colonne, normalize='columns')
                    print(contingency_percent)
            print("p value: %.10f" % p)

        except:
            print("p calculation failed")

        score = score_contingency(contingency, data, colonne)

        results = results.append(pd.DataFrame([[colonne, p, score]], columns=['Column', 'p', 'score']))

        if verbose:
            input("Appuyer sur Entrée pour continuer...")

    results = results.sort_values(by=['score'])
    for colonne, p, score in results.values:
        count = data[colonne].count()
        print(f"{colonne:40} p:{p:e}  Data count: {count:10} Score:{score:.2f}")


def normalize_int(data, column_name):
    column = data[column_name]
    min = data[column_name].min()
    max = data[column_name].max()
    column = (column - min) / (max - min)
    data[column_name] = column
    return data


def plotStackedHist(classes, labels, data, X, labelNames, ylabel='', title=''):
    voters = np.zeros((len(classes), len(labels)))
    for i, label in enumerate(labels):
        for j, votingClass in enumerate(classes):
            count = data[(X == label) & (data["vote"] == votingClass)].count()[0]
            voters[j, i] = count

    fig, ax = plt.subplots()
    bottom = voters[0] * 0
    displayLabels = labelNames[:]
    reduceLabels(displayLabels)
    for i, classByLabel in enumerate(voters):
        ax.bar(displayLabels, classByLabel, label=classes[i], bottom=bottom)
        bottom = bottom + classByLabel

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.2)
    plt.show()


def reduceLabels(labels):
    for i, label in enumerate(labels):
        listOfWords = str(label).split()
        newWord = ""
        for word in listOfWords:
            newWord = newWord + word[:8] + "\n"
        labels[i] = newWord[:-1]


def reduceLabelsAttributes(labels, length=8):
    for i, label in enumerate(labels):
        if (label[:6] == "cps19_") or (label[:6] == "pes19_"):
            label = label[6:]
        for lines in range(i % 3):
            label = "\n" + label


def standardStackedHist(data, attribute):
    attribute_vote = [attribute[:]]
    attribute_vote.append("vote")
    filteredData = data[pd.notnull(data[attribute])][attribute_vote]

    y = filteredData["vote"]
    y = y.to_numpy()
    classes = np.unique(y)

    X = filteredData[attribute]
    X = X.to_numpy()
    labels = np.unique(X)

    plotStackedHist(classes, labels, filteredData, X, labels, ylabel='Nombre de répondants',
                    title='Choix de vote: ' + attribute)


def attributeClustering(data, attribute, n_clusters=9):
    attribute_vote = [attribute[:]]
    attribute_vote.append("vote")
    filteredData = data[pd.notnull(data[attribute])][attribute_vote]

    X = filteredData[attribute]
    X = X.to_numpy()
    y = filteredData["vote"]
    y = y.to_numpy()
    kmeans = KMeans(n_clusters=n_clusters).fit(X.reshape(-1, 1))
    X2 = kmeans.predict(X.reshape(-1, 1))

    labels = np.unique(X2)
    labelNames = []
    X_classes = np.array([i for i in range(101)])
    X_predicted = kmeans.predict(X_classes.reshape(-1, 1))
    for label in labels:
        minValue = min(X_classes[X_predicted == label])
        maxValue = max(X_classes[X_predicted == label])
        labelName = "[" + str(minValue) + ",\n" + str(maxValue) + "]"
        labelNames.append(labelName)

    classes = np.unique(y)
    plotStackedHist(classes, labels, filteredData, X2, labelNames, ylabel='Nombre de répondants',
                    title='Clustering: ' + attribute)


def plotPieChart(data, attribute, title="Title"):
    filteredData = data[pd.notnull(data[attribute])][attribute]

    X = filteredData
    X = X.to_numpy()
    labels = np.unique(X)

    counts = []
    for label in labels:
        count = filteredData[filteredData == label].count()
        counts.append(count)

    fig, ax = plt.subplots()
    ax.pie(counts, radius=3, center=(4, 4), labels=labels,
           wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    plt.legend()
    plt.show()


# Fonction pas du tout nettoyée, retirée du main
def plotMultiClustering(data):
    plotPieChart(data, "vote")
    train_data = data[pd.notnull(data["vote"])]
    standardStackedHist(train_data, "cps19_issue_handle_1")

    attribut = ["cps19_party_rating_23", "cps19_party_rating_24", "cps19_party_rating_25", "cps19_party_rating_26",
                "cps19_party_rating_27", "cps19_party_rating_28", "cps19_lead_rating_23", "cps19_lead_rating_24",
                "cps19_lead_rating_25", "cps19_lead_rating_26", "cps19_lead_rating_27", "cps19_lead_rating_28",
                "cps19_cand_rating_23", "cps19_cand_rating_24", "cps19_cand_rating_25", "cps19_cand_rating_26",
                "cps19_cand_rating_27", "cps19_cand_rating_28"]
    attribut = ["cps19_lead_int_113", "cps19_lead_int_114", "cps19_lead_int_115", "cps19_lead_int_116",
                "cps19_lead_int_117", "cps19_lead_int_118", "cps19_lead_int_119", "cps19_lead_int_120",
                "cps19_lead_strong_113", "cps19_lead_strong_114", "cps19_lead_strong_115",
                "cps19_lead_strong_116", "cps19_lead_strong_117", "cps19_lead_strong_118",
                "cps19_lead_strong_119", "cps19_lead_strong_120",
                "cps19_lead_trust_113", "cps19_lead_trust_114", "cps19_lead_trust_115", "cps19_lead_trust_116",
                "cps19_lead_trust_117", "cps19_lead_trust_118", "cps19_lead_trust_119", "cps19_lead_trust_120",
                "cps19_lead_cares_113", "cps19_lead_cares_114", "cps19_lead_cares_115", "cps19_lead_cares_116",
                "cps19_lead_cares_117", "cps19_lead_cares_118", "cps19_lead_cares_119", "cps19_lead_cares_120"]

    attribut_vote = attribut[:]
    attribut_vote.append("vote")

    completeNan = False
    if completeNan:
        test = pd.notnull(data["vote"])
        testatt = test & False
        for att in attribut:
            testatt = testatt | pd.notnull(data[att])
        test = test & testatt

        extractedData = data[test][attribut_vote]
        extractedData = extractedData.fillna(0)
    else:
        test = pd.notnull(data["vote"])
        for att in attribut:
            test = test & pd.notnull(data[att])
        extractedData = data[test][attribut_vote]

    X = extractedData[attribut]
    X = X.to_numpy()
    y = extractedData["vote"]
    y = y.to_numpy()
    kmeans = KMeans(n_clusters=9).fit(X)
    X2 = kmeans.predict(X)

    labels = np.unique(X2)
    labelNames = labels
    # labelNames = []
    # X_classes = np.array([i for i in range(101)])
    # X_predicted = kmeans.predict(X_classes.reshape(-1, 1))
    # for label in labels:
    #    minValue = min(X_classes[X_predicted == label])
    #    maxValue = max(X_classes[X_predicted == label])
    #    labelName = "[" + str(minValue) + ",\n" + str(maxValue) + "]"
    #    labelNames.append(labelName)

    classes = np.unique(y)

    plotStackedHist(classes, labels, extractedData, X2, labelNames, ylabel='Répondants par groupe',
                    title='Clustering: ratings')


def fillMissingValues(data, columnList, objectList):
    for column in columnList:
        replacement = -1

        try:
            if data[pd.notnull(data[column])][column].size == 0:
                continue

            singleValue = data[pd.notnull(data[column])][column].values[0]
            datatype = type(singleValue)
            if datatype is str or column in objectList:
                replacement = "No Answer"
            else:
                replacement = data[column].median()

            data[column].fillna(replacement, inplace=True)

            if datatype is datatype:
                data[column] = np.where(data[column] == float('inf'), replacement, data[column])
                data[column] = np.where(data[column] == float('-inf'), replacement, data[column])
                data[column] = np.where(data[column] == float('nan'), replacement, data[column])
        except:
            print("{} not filled".format(column))


def addVoteColumn(data, shortClasses=True):
    vote = np.where(pd.isna(data["cps19_votechoice"]), data["cps19_votechoice_pr"], data["cps19_votechoice"])
    vote = np.where(pd.isna(vote), data["cps19_vote_unlikely"], vote)
    vote = np.where(pd.isna(vote), data["cps19_vote_unlike_pr"], vote)
    vote = np.where(pd.isna(vote), data["cps19_v_advance"], vote)
    vote = np.where(pd.isna(vote), data["cps19_vote_lean"], vote)
    vote = np.where(pd.isna(vote), data["cps19_vote_lean_pr"], vote)
    vote = np.where(vote != "I do not intend to vote", vote, None)
    if shortClasses:
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
        if column not in X.columns:
            continue
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


def combineColumns(data, columnList):
    mainColumn = columnList[0]
    for column in columnList:
        data[mainColumn] = np.where(pd.isna(data[mainColumn]), data[column], data[mainColumn])
    return data


def printConfusionMatrix(cm, classes):
    accuracy = np.array([])
    precision = np.array([])
    rappel = np.array([])
    f1score = np.array([])
    for i, aClass in enumerate(classes):
        total = np.sum(cm)
        TP = cm[i, i]
        FN = np.sum(cm[:][i]) - TP
        FP = np.sum([line[i] for line in cm]) - TP
        TN = total - TP - FN - FP
        A = (TP + TN) / total
        P = 0
        R = 0
        F1 = 0
        if TP != 0:
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
        accuracy = np.append(accuracy, A)
        precision = np.append(precision, P)
        rappel = np.append(rappel, R)
        f1score = np.append(f1score, F1)

    for i, aClass in enumerate(classes):
        print("Classe: {} \tExactitude: {:.3f} \tPrécision: {:.3f} \t Rappel: {:.3f} \tF-Score: {:.3f}".format
              (aClass, accuracy[i], precision[i], rappel[i], f1score[i]))


def main():
    data = pd.read_csv("CES19.csv")
    data = addVoteColumn(data)

    DATE_COLUMNS = ["cps19_StartDate", "cps19_EndDate"]
    COLUMNS_STRING_TO_INT = ['cps19_imm', 'cps19_refugees', 'cps19_govt_confusing', 'cps19_govt_say', 'cps19_pol_eth',
                             'cps19_lib_promises', 'cps19_snclav', 'cps19_news_cons', 'cps19_volunteer',
                             'cps19_duty_choice', 'cps19_quebec_sov', 'cps19_own_fin_retro', 'cps19_citizenship',
                             'cps19_gender', 'cps19_province', 'cps19_education', 'cps19_demsat',
                             'cps19_outcome_most', 'cps19_imp_iss_party', 'cps19_fed_id', 'cps19_imp_loc_iss_p',
                             'cps19_fed_member', 'cps19_fed_gov_sat', 'cps19_prov_gov_sat',
                             'cps19_prov_id', 'cps19_vote_2015', 'cps19_v_likely', 'cps19_spend_educ',
                             'cps19_spend_env', 'cps19_spend_just_law', 'cps19_spend_defence', 'cps19_spend_imm_min',
                             'cps19_pos_fptp', 'cps19_pos_life', 'cps19_pos_cannabis', 'cps19_pos_carbon',
                             'cps19_pos_energy', 'cps19_pos_envreg', 'cps19_pos_jobs', 'cps19_pos_subsid',
                             'cps19_pos_trade', 'cps19_econ_retro', 'cps19_econ_fed', 'cps19_ownfinanc_fed',
                             'cps19_issue_handle_1', 'cps19_issue_handle_2', 'cps19_issue_handle_3',
                             'cps19_issue_handle_4', 'cps19_issue_handle_5', 'cps19_issue_handle_6',
                             'cps19_issue_handle_7', 'cps19_issue_handle_8', 'cps19_outcome_least', 'cps19_religion',
                             'cps19_rel_imp', 'cps19_ethnicity_23', 'cps19_sexuality', 'cps19_language_69',
                             'cps19_employment', 'cps19_sector', 'cps19_union', 'cps19_children', 'cps19_income_cat',
                             'cps19_marital']
    COLUMNS_FOR_CLASSIFICATION = ['cps19_imm', 'cps19_refugees', 'cps19_govt_confusing', 'cps19_govt_say',
                                  'cps19_pol_eth', 'cps19_lib_promises', 'cps19_snclav', 'cps19_news_cons',
                                  'cps19_volunteer', 'cps19_duty_choice', 'cps19_quebec_sov', 'cps19_own_fin_retro',
                                  "cps19_StartDate", "cps19_EndDate", 'cps19_citizenship', 'cps19_yob', 'cps19_gender',
                                  'cps19_province', 'cps19_education', 'cps19_demsat', 'cps19_outcome_most',
                                  'cps19_imp_iss_party', 'cps19_fed_id', 'cps19_fed_member', 'cps19_fed_gov_sat',
                                  'cps19_prov_gov_sat',
                                  'cps19_prov_id', 'cps19_vote_2015', "cps19_party_rating_23", "cps19_party_rating_24",
                                  "cps19_party_rating_25", "cps19_party_rating_26", "cps19_party_rating_27",
                                  "cps19_party_rating_28", "cps19_cand_rating_23", "cps19_cand_rating_24",
                                  "cps19_cand_rating_25", "cps19_cand_rating_26", "cps19_cand_rating_27",
                                  "cps19_cand_rating_28", "cps19_lead_rating_23", "cps19_lead_rating_24",
                                  "cps19_lead_rating_25", "cps19_lead_rating_26", "cps19_lead_rating_27",
                                  "cps19_lead_rating_28", 'cps19_imp_loc_iss_p', 'cps19_interest_gen_1',
                                  'cps19_interest_elxn_1', 'cps19_v_likely', 'cps19_lr_scale_bef_1',
                                  'cps19_lr_parties_1', 'cps19_lr_parties_2', 'cps19_lr_parties_3',
                                  'cps19_lr_parties_4', 'cps19_lr_parties_5', 'cps19_lr_parties_6',
                                  "cps19_lead_int_113", "cps19_lead_int_114", "cps19_lead_int_115",
                                  "cps19_lead_int_116", "cps19_lead_int_117", "cps19_lead_int_118",
                                  "cps19_lead_int_119", "cps19_lead_int_120", "cps19_lead_strong_113",
                                  "cps19_lead_strong_114", "cps19_lead_strong_115", "cps19_lead_strong_116",
                                  "cps19_lead_strong_117", "cps19_lead_strong_118", "cps19_lead_strong_119",
                                  "cps19_lead_strong_120", "cps19_lead_trust_113", "cps19_lead_trust_114",
                                  "cps19_lead_trust_115", "cps19_lead_trust_116", "cps19_lead_trust_117",
                                  "cps19_lead_trust_118", "cps19_lead_trust_119", "cps19_lead_trust_120",
                                  "cps19_lead_cares_113", "cps19_lead_cares_114", "cps19_lead_cares_115",
                                  "cps19_lead_cares_116", "cps19_lead_cares_117", "cps19_lead_cares_118",
                                  "cps19_lead_cares_119", "cps19_lead_cares_120", 'cps19_spend_educ', 'cps19_spend_env',
                                  'cps19_spend_just_law', 'cps19_spend_defence', 'cps19_spend_imm_min',
                                  'cps19_pos_fptp', 'cps19_pos_life', 'cps19_pos_cannabis', 'cps19_pos_carbon',
                                  'cps19_pos_energy', 'cps19_pos_envreg', 'cps19_pos_jobs', 'cps19_pos_subsid',
                                  'cps19_pos_trade', 'cps19_econ_retro', 'cps19_econ_fed', 'cps19_ownfinanc_fed',
                                  'cps19_issue_handle_1', 'cps19_issue_handle_2', 'cps19_issue_handle_3',
                                  'cps19_issue_handle_4', 'cps19_issue_handle_5', 'cps19_issue_handle_6',
                                  'cps19_issue_handle_7', 'cps19_issue_handle_8', 'cps19_most_seats_1',
                                  'cps19_most_seats_2', 'cps19_most_seats_3', 'cps19_most_seats_4',
                                  'cps19_most_seats_5', 'cps19_most_seats_6', 'cps19_win_local_1', 'cps19_win_local_2',
                                  'cps19_win_local_3', 'cps19_win_local_4', 'cps19_win_local_5', 'cps19_win_local_6',
                                  'cps19_outcome_least', 'cps19_religion', 'cps19_rel_imp', 'cps19_ethnicity_23',
                                  'cps19_sexuality', 'cps19_language_69', 'cps19_employment', 'cps19_sector',
                                  'cps19_union', 'cps19_children', 'cps19_income_cat', 'cps19_marital',
                                  'cps19_property_1', 'cps19_property_2', 'cps19_property_3',
                                  'cps19_property_4', 'cps19_property_5', 'cps19_property_6', 'cps19_household']
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
                          "cps19_lead_cares_120"],
                         ['cps19_property_1', 'cps19_property_2', 'cps19_property_3', 'cps19_property_4',
                          'cps19_property_5', 'cps19_property_6']]
    COLUMNS_TO_COMBINE = [['cps19_v_likely', 'cps19_v_likely_pr'], ['cps19_lr_scale_bef_1', 'cps19_lr_scale_aft_1'],
                          ['cps19_ethnicity_23', 'cps19_ethnicity_24', 'cps19_ethnicity_25', 'cps19_ethnicity_26',
                           'cps19_ethnicity_27', 'cps19_ethnicity_28', 'cps19_ethnicity_29', 'cps19_ethnicity_30',
                           'cps19_ethnicity_31', 'cps19_ethnicity_32', 'cps19_ethnicity_33', 'cps19_ethnicity_34',
                           'cps19_ethnicity_35', 'cps19_ethnicity_36', 'cps19_ethnicity_37', 'cps19_ethnicity_38',
                           'cps19_ethnicity_39', 'cps19_ethnicity_40', 'cps19_ethnicity_41', 'cps19_ethnicity_42',
                           'cps19_ethnicity_43'],
                          ['cps19_language_69', 'cps19_language_70', 'cps19_language_71', 'cps19_language_72',
                           'cps19_language_73', 'cps19_language_74', 'cps19_language_75', 'cps19_language_76',
                           'cps19_language_77', 'cps19_language_78', 'cps19_language_79', 'cps19_language_80',
                           'cps19_language_81', 'cps19_language_82', 'cps19_language_83', 'cps19_language_84',
                           'cps19_language_85']]

    # COLUMNS_TO_EXPLORE? : cps19_imp_iss

    COLUMNS_FOR_CLASSIFICATION_INIT = ["cps19_StartDate", 'cps19_province', 'cps19_outcome_most', 'cps19_imp_iss_party',
                                       'cps19_fed_id',
                                       'cps19_fed_member', 'cps19_fed_gov_sat',
                                       'cps19_prov_id', 'cps19_vote_2015', "cps19_party_rating_23",
                                       "cps19_party_rating_24", "cps19_party_rating_25", "cps19_party_rating_26",
                                       "cps19_party_rating_27", "cps19_party_rating_28",
                                       "cps19_lead_rating_23", "cps19_lead_rating_24", "cps19_lead_rating_25",
                                       "cps19_lead_rating_26", "cps19_lead_rating_27", "cps19_lead_rating_28"]

    COLUMNS_STRING_TO_INT = list(set(COLUMNS_STRING_TO_INT).intersection(COLUMNS_FOR_CLASSIFICATION))

    COLUMNS_GENERATED = ["cps19_StartDate", "cps19_party_rating_23", "cps19_party_rating_24", "cps19_party_rating_25", "cps19_party_rating_26", "cps19_party_rating_27", "cps19_party_rating_28", "cps19_lead_rating_23", "cps19_lead_rating_24", "cps19_lead_rating_25", "cps19_lead_rating_26", "cps19_lead_rating_27", "cps19_lead_rating_28", "cps19_fed_id_Another party (please specify)", "cps19_fed_id_Bloc Qu<e9>b<e9>cois", "cps19_fed_id_Conservative", "cps19_fed_id_Don't know/ Prefer not to answer", "cps19_fed_id_Green", "cps19_fed_id_Liberal", "cps19_fed_id_None of these", "cps19_fed_id_People's Party", "cps19_fed_id_ndp", "cps19_vote_2015_Another party (please specify)", "cps19_vote_2015_Bloc Qu<e9>b<e9>cois", "cps19_vote_2015_Conservative Party", "cps19_vote_2015_Don't know/ Prefer not to answer", "cps19_vote_2015_Green Party", "cps19_vote_2015_Liberal Party", "cps19_vote_2015_No Answer", "cps19_vote_2015_ndp", "cps19_outcome_most_Conservative majority", "cps19_outcome_most_Conservative minority", "cps19_outcome_most_Don't know/ Prefer not to answer", "cps19_outcome_most_Liberal majority", "cps19_outcome_most_Liberal minority", "cps19_outcome_most_NDP majority", "cps19_outcome_most_NDP minority", "cps19_outcome_most_No Answer", "cps19_outcome_most_Other government", "cps19_fed_member_Bloc Qu<e9>b<e9>cois", "cps19_fed_member_Conservative Party", "cps19_fed_member_Don't know/ Prefer not to answer", "cps19_fed_member_Green Party", "cps19_fed_member_Liberal Party", "cps19_fed_member_No Answer", "cps19_fed_member_Other (please specify)", "cps19_fed_member_People's Party", "cps19_fed_member_ndp", "cps19_province_Alberta", "cps19_province_British Columbia", "cps19_province_Manitoba", "cps19_province_New Brunswick", "cps19_province_Newfoundland and Labrador", "cps19_province_Northwest Territories", "cps19_province_Nova Scotia", "cps19_province_Nunavut", "cps19_province_Ontario", "cps19_province_Prince Edward Island", "cps19_province_Quebec", "cps19_province_Saskatchewan", "cps19_province_Yukon", "cps19_prov_id_Alberta Party", "cps19_prov_id_Another party (please specify)", "cps19_prov_id_Coalition Avenir Qu<e9>bec", "cps19_prov_id_Conservative", "cps19_prov_id_Don't know/ Prefer not to answer", "cps19_prov_id_Green", "cps19_prov_id_Liberal", "cps19_prov_id_No Answer", "cps19_prov_id_None of these", "cps19_prov_id_Parti Qu<e9>b<e9>cois", "cps19_prov_id_People's Alliance", "cps19_prov_id_Progressive Conservative", "cps19_prov_id_Qu<e9>bec Solidaire", "cps19_prov_id_Saskatchewan Party", "cps19_prov_id_United Conservative", "cps19_prov_id_Yukon Party", "cps19_prov_id_ndp", "cps19_fed_gov_sat_Don't know/ Prefer not to answer", "cps19_fed_gov_sat_Fairly satisfied", "cps19_fed_gov_sat_Not at all satisfied", "cps19_fed_gov_sat_Not very satisfied", "cps19_fed_gov_sat_Very satisfied", "cps19_imp_iss_party_Another party (please specify)", "cps19_imp_iss_party_Bloc Qu<e9>b<e9>cois", "cps19_imp_iss_party_Conservative Party", "cps19_imp_iss_party_Don't know/ Prefer not to answer", "cps19_imp_iss_party_Green Party", "cps19_imp_iss_party_Liberal Party", "cps19_imp_iss_party_No Answer", "cps19_imp_iss_party_People's Party", "cps19_imp_iss_party_ndp"]
    COLUMNS_GENERATED2 = ["cps19_StartDate", "cps19_party_rating_23", "cps19_party_rating_24", "cps19_party_rating_25", "cps19_party_rating_27", "cps19_lead_rating_24", "cps19_lead_rating_25", "cps19_lead_rating_28", "cps19_fed_id_Another party (please specify)", "cps19_fed_id_Bloc Qu<e9>b<e9>cois", "cps19_fed_id_Don't know/ Prefer not to answer", "cps19_fed_id_Green", "cps19_fed_id_Liberal", "cps19_fed_id_People's Party", "cps19_vote_2015_Another party (please specify)", "cps19_vote_2015_Bloc Qu<e9>b<e9>cois", "cps19_vote_2015_Conservative Party", "cps19_vote_2015_Don't know/ Prefer not to answer", "cps19_vote_2015_Green Party", "cps19_vote_2015_Liberal Party", "cps19_vote_2015_ndp", "cps19_outcome_most_Conservative majority", "cps19_outcome_most_Don't know/ Prefer not to answer", "cps19_outcome_most_Liberal majority", "cps19_outcome_most_NDP majority", "cps19_outcome_most_NDP minority", "cps19_fed_member_Don't know/ Prefer not to answer", "cps19_fed_member_Green Party", "cps19_fed_member_Liberal Party", "cps19_fed_member_Other (please specify)", "cps19_fed_member_People's Party", "cps19_fed_member_ndp", "cps19_province_Alberta", "cps19_province_British Columbia", "cps19_province_Manitoba", "cps19_province_New Brunswick", "cps19_province_Newfoundland and Labrador", "cps19_province_Northwest Territories", "cps19_province_Nova Scotia", "cps19_province_Ontario", "cps19_province_Prince Edward Island", "cps19_province_Yukon", "cps19_prov_id_Alberta Party", "cps19_prov_id_Another party (please specify)", "cps19_prov_id_Coalition Avenir Qu<e9>bec", "cps19_prov_id_Conservative", "cps19_prov_id_Don't know/ Prefer not to answer", "cps19_prov_id_Green", "cps19_prov_id_Liberal", "cps19_prov_id_No Answer", "cps19_prov_id_Progressive Conservative", "cps19_prov_id_Qu<e9>bec Solidaire", "cps19_prov_id_Saskatchewan Party", "cps19_prov_id_United Conservative", "cps19_prov_id_Yukon Party", "cps19_fed_gov_sat_Don't know/ Prefer not to answer", "cps19_fed_gov_sat_Not at all satisfied", "cps19_fed_gov_sat_Not very satisfied", "cps19_fed_gov_sat_Very satisfied", "cps19_imp_iss_party_Another party (please specify)", "cps19_imp_iss_party_Bloc Qu<e9>b<e9>cois", "cps19_imp_iss_party_Conservative Party", "cps19_imp_iss_party_Don't know/ Prefer not to answer", "cps19_imp_iss_party_Liberal Party", "cps19_imp_iss_party_No Answer", "cps19_imp_iss_party_People's Party", "cps19_imp_iss_party_ndp"]


    #data['total_no_answer'] = (data[COLUMNS_FOR_CLASSIFICATION] == "Don't know/ Prefer not to answer").sum(axis=1)
    #COLUMNS_FOR_CLASSIFICATION.append('total_no_answer')

    for column in DATE_COLUMNS:
        data = dateToInt(data, column)

    for group in BOOLEANS_TO_CLEAN:
        data = clean_boolean(data, group)

    for group in COLUMNS_TO_COMBINE:
        data = combineColumns(data, group)

    fillMissingValues(data, COLUMNS_FOR_CLASSIFICATION, COLUMNS_STRING_TO_INT)

    X_all = data[COLUMNS_FOR_CLASSIFICATION]
    unknown_indices = util.readResultFile()
    X_unknown = data.iloc[unknown_indices][COLUMNS_FOR_CLASSIFICATION]

    usingLabelEncoder = False
    if (usingLabelEncoder):
        le = preprocessing.LabelEncoder()
        for column in COLUMNS_STRING_TO_INT:
            le.fit(data[column])
            X_all[column] = le.transform(X_all[column])
    else:
        stringToOneHot(X_all, X_unknown, COLUMNS_STRING_TO_INT)

    X = X_all[pd.notnull(data['vote'])]
    y = data[pd.notnull(data['vote'])]['vote']

    if True:
        column_list = COLUMNS_GENERATED
        while True:
            column_list = util.removeColumns(X, y, column_list)
            column_list = util.addColumns(X, y, column_list)



    if False:  # Add columns one by one
        best_score = float('inf')

        column_list = COLUMNS_GENERATED

        for i in range(3):
            X_filtered = X[column_list]
            X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.1)
            clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            if score < best_score:
                best_score = score

        for i in range(100):
            best_column = None
            count = 0
            X_columns = [col for col in X.columns]
            random.shuffle(X_columns)
            for column in X_columns:
                if column in column_list:
                    continue
                columns = column_list.copy()
                columns.append(column)
                X_filtered = X[columns]

                X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.9)
                clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
                clf.fit(X_train, y_train)

                score = clf.score(X_test, y_test)
                if score > best_score:
                    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.9)
                    clf = RandomForestClassifier(n_jobs=-1)
                    clf.fit(X_train, y_train)
                    score = min(clf.score(X_test, y_test), score)
                    if score > best_score:
                        best_score = score
                        best_column = column
                        print("\t{}: New best! Column: {}   \tScore: {}".format(i, best_column, best_score))
                count += 1
                if count >= 10 and not (best_column is None):
                    break

            if best_column is None:
                break

            column_list.append(best_column)
            print("{}: Testing complete! Column: {}   \tScore: {}".format(i, best_column, best_score))
        print("Final list of columns:")
        print("[", end='')
        for column in column_list:
            print('"{}", '.format(column), end='')
        print("]")
        return

    if (False):  # Remove columns one by one
        best_score = float('inf')

        column_list = [col for col in X.columns]

        for i in range(3):
            X_filtered = X[column_list]
            X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.1)
            clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            if score < best_score:
                best_score = score

        for i in range(100):
            best_column = None
            count = 0
            X_columns = [col for col in column_list]
            random.shuffle(X_columns)
            for column in X_columns:
                columns = [col for col in column_list]
                columns.remove(column)
                X_filtered = X[columns]

                X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.1)
                clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
                clf.fit(X_train, y_train)

                score = clf.score(X_test, y_test)
                if score > best_score:
                    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.1)
                    clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
                    clf.fit(X_train, y_train)
                    score = min(clf.score(X_test, y_test), score)
                    if score > best_score:
                        best_score = score
                        best_column = column
                        print("\t{}: New best! Column: {}   \tScore: {}".format(i, best_column, best_score))
                        count = 0
                count += 1
                if count >= 10 and not (best_column is None):
                    break

            if best_column is None:
                break
            column_list.remove(best_column)
            print("{}: Testing complete! Column: {}   \tScore: {}".format(i, best_column, best_score))

        print("Final list of columns:")
        print("[", end='')
        for column in column_list:
            print('"{}", '.format(column), end='')
        return
        print("]")

    if False:  # Loop to find parameters
        results = []
        for imaxDepth in range(1, 8):
            for iminLeaf in range(0, 8):
                for iNtrees in range(0, 8):
                    scores = []
                    if imaxDepth <= 0:
                        maxDepth = None
                    else:
                        maxDepth = 2 ** imaxDepth
                    minLeaf = 2 ** iminLeaf
                    n_est = 2 ** iNtrees

                    startTime = time.time()
                    for i in range(5):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

                        clf = RandomForestClassifier(max_depth=maxDepth, min_samples_leaf=minLeaf, n_estimators=n_est, n_jobs=-1)
                        clf.fit(X_train, y_train)

                        score = clf.score(X_test, y_test)
                        scores.append(score)

                        if False:
                            predictions = clf.predict(X_test)
                            cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                                          display_labels=clf.classes_)
                            disp.plot()
                            plt.show()

                            predictions_prob = clf.predict_proba(X_test)
                            badX = X_test[predictions != y_test]
                            # util.writeResultFile(X_unknown, clf)
                    avgScore = np.array(scores).mean()
                    totalTime = time.time() - startTime
                    print("Max Depth: {}\tMin Leaf: {}\tn Trees: {}\tScore: {}\tScore: {}".format(maxDepth, minLeaf, n_est, avgScore, totalTime))
                    results.append([maxDepth, minLeaf, n_est, avgScore, totalTime])
        scores = [i[3] for i in results]
        ok_score = best_score * 0.9925
        index = [i > ok_score for i in scores]
        results = np.array(results)
        for i in results[index]:
            print("{} & {} & {} & {} \\\\".format(i[0], i[1], i[2], i[3]))
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
    disp.plot()
    plt.title("Forêt aléatoire - Score: {}".format(score))
    plt.show()

    printConfusionMatrix(cm, clf.classes_)

    export_graphviz(clf.estimators_[0], out_file='tree.dot',
                    feature_names=X.columns,
                    class_names=clf.classes_,
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=1200'])

    newData = pd.DataFrame(data, columns=['vote'])
    predictions_prob = clf.predict_proba(X_all)
    classes = clf.classes_
    columns = []
    for i, aClass in enumerate(classes):
        newData['predict' + aClass] = [prediction[i] for prediction in predictions_prob]
        columns.append('predict' + aClass)

        classY = np.where(y_train == aClass, y_train, 'other')

        weights = {aClass: np.sum(classY == aClass), 'other': np.sum(classY != aClass)}
        classclf = RandomForestClassifier(class_weight=weights, max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
        classclf.fit(X_train, classY)
        classPredictions = classclf.predict_proba(X_all)
        for i, anotherClass in enumerate(classclf.classes_):
            if aClass == anotherClass:
                newData['classPredict' + aClass] = [prediction[i] for prediction in classPredictions]
                columns.append('classPredict' + aClass)

    train_indices = X_train.index
    test_indices = X_test.index

    bayesianClf = ComplementNB()
    X_train_bayes = newData.iloc[train_indices][columns]
    y_train_bayes = newData.iloc[train_indices]['vote']
    X_test_bayes = newData.iloc[test_indices][columns]
    y_test_bayes = newData.iloc[test_indices]['vote']
    bayesianClf.fit(X_train_bayes, y_train_bayes)

    score = bayesianClf.score(X_test_bayes, y_test_bayes)
    predictionsBayes = bayesianClf.predict(X_test_bayes)
    cm = confusion_matrix(y_test_bayes, predictionsBayes, labels=bayesianClf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=bayesianClf.classes_)
    disp.plot()
    plt.title("Bayesian - Score: {}".format(score))
    plt.show()

    f_imp = clf.feature_importances_
    feat = clf.feature_names_in_
    ordre = np.argsort(f_imp)
    f_imp = f_imp[ordre]
    feat = feat[ordre]
    for f, imp in zip(feat, f_imp):
        print("{}: {}".format(f, imp))

    comparison = predictions == y_test
    bad_sample = X_test.loc[[X_test[comparaison].index[0]]]
    bad_class = y_test.loc[bad_sample.index].values
    bad_prediction = clf.predict(bad_sample)


if __name__ == '__main__':
    main()
