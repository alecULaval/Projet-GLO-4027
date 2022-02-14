import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
import os

print(pd.__version__)

desired_width = 1080
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 60)

VOTE_QUESTIONS = ["cps19_votechoice","cps19_votechoice_pr","cps19_vote_unlikely","cps19_vote_unlike_pr",
                  "cps19_v_advance","cps19_votechoice_7_TEXT","cps19_vote_unlikely_7_TEXT","cps19_vote_unlike_pr_7_TEXT",
                  "cps19_v_advance_7_TEXT", "cps19_vote_lean","cps19_vote_lean_7_TEXT", "cps19_vote_lean_pr",
                  "cps19_vote_lean_pr_7_TEXT", "cps19_2nd_choice","cps19_2nd_choice_7_TEXT", "cps19_2nd_choice_pr",
                  "cps19_2nd_choice_pr_7_TEXT", "cps19_not_vote_for_1","cps19_not_vote_for_2", "cps19_not_vote_for_3",
                  "cps19_not_vote_for_4", "cps19_not_vote_for_5","cps19_not_vote_for_6", "cps19_not_vote_for_7",
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
            "cps19_lead_int_114", "cps19_lead_int_115", "cps19_lead_int_116", "cps19_lead_int_117", "cps19_lead_int_118"]]
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
    min_responses = total_responses/1000
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
        print(f"{column:40} {answer:40} {vote:40} Score:{score:.2f} Size:{size:.2f}")


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
        listOfWords = label.split()
        newWord = ""
        for word in listOfWords:
            newWord = newWord + word[:8] + "\n"
        labels[i] = newWord[:-1]

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
    plotStackedHist(classes, labels, filteredData, X2, labelNames, ylabel='Nombre de répondants', title='Clustering: ' + attribute)


def main():
    data = pd.read_csv("CES19.csv")
    # questionnaire(data)

    data["isUnknown"] = ((pd.isna(data["cps19_votechoice"])) & (pd.isna(data["cps19_votechoice_pr"]))
         & (pd.isna(data["cps19_vote_unlikely"])) & (pd.isna(data["cps19_vote_unlike_pr"]))
         & (pd.isna(data["cps19_v_advance"])) & (pd.isna(data["cps19_votechoice_7_TEXT"]))
         & (pd.isna(data["cps19_vote_unlikely_7_TEXT"])) & (pd.isna(data["cps19_vote_unlike_pr_7_TEXT"]))
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
    data["vote"] = vote
    data = clean_boolean(data, ["cps19_lead_int_113", "cps19_lead_int_114", "cps19_lead_int_115", "cps19_lead_int_116",
                                "cps19_lead_int_117", "cps19_lead_int_118", "cps19_lead_int_119", "cps19_lead_int_120"])
    data = clean_boolean(data, ["cps19_lead_strong_113", "cps19_lead_strong_114", "cps19_lead_strong_115",
                                "cps19_lead_strong_116", "cps19_lead_strong_117", "cps19_lead_strong_118",
                                "cps19_lead_strong_119", "cps19_lead_strong_120"])
    data = clean_boolean(data, ["cps19_lead_trust_113", "cps19_lead_trust_114", "cps19_lead_trust_115", "cps19_lead_trust_116",
                                "cps19_lead_trust_117", "cps19_lead_trust_118", "cps19_lead_trust_119", "cps19_lead_trust_120"])
    data = clean_boolean(data, ["cps19_lead_cares_113", "cps19_lead_cares_114", "cps19_lead_cares_115", "cps19_lead_cares_116",
                                "cps19_lead_cares_117", "cps19_lead_cares_118", "cps19_lead_cares_119", "cps19_lead_cares_120"])
    #exploration(data, verbose=False)
    #get_best_questions(data)

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

    attribut = ["cps19_interest_gen_1"]
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
    #labelNames = []
    #X_classes = np.array([i for i in range(101)])
    #X_predicted = kmeans.predict(X_classes.reshape(-1, 1))
    #for label in labels:
    #    minValue = min(X_classes[X_predicted == label])
    #    maxValue = max(X_classes[X_predicted == label])
    #    labelName = "[" + str(minValue) + ",\n" + str(maxValue) + "]"
    #    labelNames.append(labelName)


    classes = np.unique(y)

    plotStackedHist(classes, labels, extractedData, X2, labelNames, ylabel='Répondants par groupe', title='Clustering')

    print("Break Here")

if __name__ == '__main__':
    main()
