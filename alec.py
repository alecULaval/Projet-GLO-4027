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
from datetime import datetime
import util
import math




Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
Cols = ['A', 'B', 'C', 'D']
df = pd.DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)

sns.heatmap(df, annot=True)
#plt.show()


data = pd.read_csv("CES19.csv")
data['finalVote'] = 'A'
columns = data.columns


def findFinalVotes(actualVotes):
    realVote = pd.NA
    count = 0
    for i in range(0, len(actualVotes)):

        if not pd.isna(actualVotes.iloc[i]['cps19_votechoice']):
            realVote = actualVotes.iloc[i]['cps19_votechoice']
        elif not pd.isna(actualVotes.iloc[i]['cps19_votechoice_pr']):
            realVote = actualVotes.iloc[i]['cps19_votechoice_pr']
        elif not pd.isna(actualVotes.iloc[i]['cps19_vote_unlikely']):
            realVote = actualVotes.iloc[i]['cps19_vote_unlikely']
        elif not pd.isna(actualVotes.iloc[i]['cps19_vote_unlike_pr']):
            realVote = actualVotes.iloc[i]['cps19_vote_unlike_pr']
        elif not pd.isna(actualVotes.iloc[i]['cps19_v_advance']):
            realVote = actualVotes.iloc[i]['cps19_v_advance']
        else:
            #print(actualVotes.iloc[i]['cps19_votechoice'], actualVotes.iloc[i]['cps19_votechoice_pr'], actualVotes.iloc[i]['cps19_vote_unlikely'],actualVotes.iloc[i]['cps19_vote_unlike_pr'], actualVotes.iloc[i]['cps19_v_advance'])
            #print(i, actualVotes.iloc[i]['Unnamed: 0'], actualVotes.iloc[i]['Unnamed: 0'])
            count +=1
        actualVotes.iloc[i]['finalVote'] = realVote
    return actualVotes

def findCorrelationWithQuebecProvince(provinceAnswers, finalVotes):
    provinceAnsweredCount = 0
    quebecResidentTotalCount = 0
    quebecResidentVotedForBlocCount = 0
    blocVoterNotQuebecResidentCount =0
    for i in range(0, len(provinceAnswers)):
        #print(finalVotes.iloc[i]['finalVote'])
        if not pd.isna(provinceAnswers.iloc[i]['cps19_province']):
            provinceAnsweredCount +=1

        if provinceAnswers.iloc[i]['cps19_province'] == 'Quebec':
            quebecResidentTotalCount +=1

        if provinceAnswers.iloc[i]['cps19_province'] == 'Quebec' and finalVotes.iloc[i]['finalVote'] == "Bloc Qu<e9>b<e9>cois":
            quebecResidentVotedForBlocCount +=1

        if provinceAnswers.iloc[i]['cps19_province'] != 'Quebec' and finalVotes.iloc[i]['finalVote'] == "Bloc Qu<e9>b<e9>cois":
            if not pd.isna(provinceAnswers.iloc[i]['cps19_province']):
                #print(provinceAnswers.iloc[i]['Unnamed: 0'], provinceAnswers.iloc[i]['pes19_province'] , finalVotes.iloc[i]['finalVote'])
                blocVoterNotQuebecResidentCount +=1

    return [provinceAnsweredCount, quebecResidentTotalCount, quebecResidentVotedForBlocCount, blocVoterNotQuebecResidentCount]

def findCorrelationForBiggestIssue(bestAdressesIssue, finalVotes):
    bestAdressesPartyTotalCount = 0
    bestAdressesIssueSameForVoteCount = 0
    bestAdressesIssueDifferentForVoteCount = 0
    for i in range(0, len(bestAdressesIssue)):
        if not pd.isna(bestAdressesIssue.iloc[i]['cps19_imp_iss_party']) and not pd.isna(finalVotes.iloc[i]['finalVote']):
            bestAdressesPartyTotalCount += 1

        if bestAdressesIssue.iloc[i]['cps19_imp_iss_party'] == finalVotes.iloc[i]['finalVote']\
                and (not pd.isna(bestAdressesIssue.iloc[i]['cps19_imp_iss_party']) and not pd.isna(finalVotes.iloc[i]['finalVote'])):
                #print( "Best adresses: {bestAdresses}  Final Vote: {finalVote}".format(
                        #bestAdresses=bestAdressesIssue.iloc[i]['cps19_imp_iss_party'], finalVote=finalVotes.iloc[i]['finalVote']))
                bestAdressesIssueSameForVoteCount +=1

        if bestAdressesIssue.iloc[i]['cps19_imp_iss_party'] != finalVotes.iloc[i]['finalVote']\
                and (not pd.isna(bestAdressesIssue.iloc[i]['cps19_imp_iss_party']) and not pd.isna(finalVotes.iloc[i]['finalVote'])):

                #print( "Best adresses: {bestAdresses}  Final Vote: {finalVote}".format(
                        #bestAdresses=bestAdressesIssue.iloc[i]['cps19_imp_iss_party'], finalVote=finalVotes.iloc[i]['finalVote']))
                bestAdressesIssueDifferentForVoteCount +=1

    return [bestAdressesPartyTotalCount, bestAdressesIssueSameForVoteCount, bestAdressesIssueDifferentForVoteCount]

def findCorrelationMostWantedOutcome(desiredOutcome, finalVotes):
    bestOutcomeTotalCount = 0
    desiredOutcomeSameThenVoteLiberals =0
    desiredOutcomeDifferentThenVoteLiberals = 0
    desiredOutcomeSameThenVoteConservative =0
    desiredOutcomeDifferentThenVoteConservative = 0
    desiredOutcomeSameThenVoteNDP =0
    desiredOutcomeDifferentThenVoteNDP = 0

    for i in range(0, len(desiredOutcome)):
        if not pd.isna(desiredOutcome.iloc[i]['cps19_outcome_most']) and not pd.isna(finalVotes.iloc[i]['finalVote']):
            #print( "Best adresses: {bestOutcome}  Final Vote: {finalVote}".format(
            #bestOutcome=desiredOutcome.iloc[i]['cps19_outcome_most'], finalVote=finalVotes.iloc[i]['finalVote']))
            bestOutcomeTotalCount += 1

        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "Liberal majority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "Liberal minority")\
                and finalVotes.iloc[i]['finalVote'] == "Liberal Party":
            #print( "Best adresses: {bestOutcome}  Final Vote: {finalVote}".format(
            #bestOutcome=desiredOutcome.iloc[i]['cps19_outcome_most'], finalVote=finalVotes.iloc[i]['finalVote']))
            desiredOutcomeSameThenVoteLiberals +=1


        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "Liberal majority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "Liberal minority")\
                and finalVotes.iloc[i]['finalVote'] != "Liberal Party":
            desiredOutcomeDifferentThenVoteLiberals +=1

        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "Conservative majority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "Conservative minority")\
                and finalVotes.iloc[i]['finalVote'] == "Conservative Party":
            #print( "Best adresses: {bestOutcome}  Final Vote: {finalVote}".format(
            #bestOutcome=desiredOutcome.iloc[i]['cps19_outcome_most'], finalVote=finalVotes.iloc[i]['finalVote']))
            desiredOutcomeSameThenVoteConservative +=1


        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "Conservative majority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "Conservative minority")\
                and finalVotes.iloc[i]['finalVote'] != "Conservative Party":
            desiredOutcomeDifferentThenVoteConservative +=1

        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "NDP minority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "NDP majority")\
                and finalVotes.iloc[i]['finalVote'] == "ndp":
            #print( "Best adresses: {bestOutcome}  Final Vote: {finalVote}".format(
            #bestOutcome=desiredOutcome.iloc[i]['cps19_outcome_most'], finalVote=finalVotes.iloc[i]['finalVote']))
            desiredOutcomeSameThenVoteNDP +=1


        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "NDP minority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "NDP majority")\
                and finalVotes.iloc[i]['finalVote'] != "ndp":
            desiredOutcomeDifferentThenVoteNDP +=1

    return [bestOutcomeTotalCount,desiredOutcomeSameThenVoteLiberals, desiredOutcomeDifferentThenVoteLiberals,
            desiredOutcomeSameThenVoteConservative, desiredOutcomeDifferentThenVoteConservative,
            desiredOutcomeSameThenVoteNDP, desiredOutcomeDifferentThenVoteNDP]

def findCorrelationAffiliationPgilosophique(affinitePolitique, confidenceAffinete, finalVotes):
    nbrSame = 0
    nbrDifferent = 0
    notConfident = 0
    semiConfident = 0
    confident = 0

    notConfidentIfSame = 0
    semiConfidentIfSame = 0
    confidentIfSame = 0

    for i in range(0, len(affinitePolitique)):

        if (affinitePolitique.iloc[i]['cps19_fed_id'] == finalVotes.iloc[i]['finalVote'] or (
                affinitePolitique.iloc[i]['cps19_fed_id'] == "Liberal" and finalVotes.iloc[i]['finalVote'] == "Liberal Party")
                or (affinitePolitique.iloc[i]['cps19_fed_id'] == "Conservative" and finalVotes.iloc[i]['finalVote'] == "Conservative Party")
                or (affinitePolitique.iloc[i]['cps19_fed_id'] == "Green" and finalVotes.iloc[i]['finalVote'] == "Green Party")):
            nbrSame += 1
            if (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Not very strongly"):
                notConfidentIfSame += 1
            elif (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Fairly strongly"):
                semiConfidentIfSame += 1
            elif (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Very strongly"):
                confidentIfSame += 1

        else:
            if not pd.isna(affinitePolitique.iloc[i]['cps19_fed_id']) and not pd.isna(finalVotes.iloc[i]['finalVote'])\
                    and affinitePolitique.iloc[i]['cps19_fed_id'] != "None of these" \
                    and affinitePolitique.iloc[i]['cps19_fed_id'] != "Don't know/ Prefer not to answer" \
                    and affinitePolitique.iloc[i]['cps19_fed_id'] != "Another party (please specify)" :
                nbrDifferent += 1

                if (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Not very strongly"):
                    notConfident += 1
                elif (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Fairly strongly"):
                    semiConfident += 1
                elif (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Very strongly"):
                    confident += 1

                #print(
                    #"Affinite = {affinitePolitique} , vrai vote =  {voteChoice}, Confiance = {confiance}".format(
                        #affinitePolitique=affinitePolitique.iloc[i]['cps19_fed_id'], voteChoice=finalVotes.iloc[i]['finalVote'], confiance=confidenceAffinete.iloc[i]['cps19_fed_id_str']))

    pourcentageNonConfient = notConfident / (notConfident + semiConfident + confident) * 100
    pourcentageSemiConfient = semiConfident / (notConfident + semiConfident + confident) * 100
    pourcentageConfient = confident / (notConfident + semiConfident + confident) * 100

    pourcentageNonConfientIfSame = notConfidentIfSame / (
                notConfidentIfSame + semiConfidentIfSame + confidentIfSame) * 100
    pourcentageSemiConfientIfSame = semiConfidentIfSame / (
                notConfidentIfSame + semiConfidentIfSame + confidentIfSame) * 100
    pourcentageConfientIfSame = confidentIfSame / (notConfidentIfSame + semiConfidentIfSame + confidentIfSame) * 100

    print(
        "Parmi ceux qui avaient un vote different de leur affinite, {pourcentageNonConfient} % ne sont pas confiants, "
        "{pourcentageSemiConfient} % sont relativement confiants et {pourcentageConfient} % sont confiants".format(
            pourcentageNonConfient=pourcentageNonConfient, pourcentageSemiConfient=pourcentageSemiConfient,
            pourcentageConfient=pourcentageConfient))

    print(
        "Parmi ceux qui avaient LE MEME VOTE que celui de leur affinite, {pourcentageNonConfientIfSame} % ne sont pas confiants, "
        "{pourcentageSemiConfientIfSame} % sont relativement confiants et {pourcentageConfientIfSame} % sont confiants".format(
            pourcentageNonConfientIfSame=pourcentageNonConfientIfSame,
            pourcentageSemiConfientIfSame=pourcentageSemiConfientIfSame,
            pourcentageConfientIfSame=pourcentageConfientIfSame))

    sucessRate = nbrSame / (nbrSame + nbrDifferent) * 100
    print(nbrSame + nbrDifferent)
    print(sucessRate)

    return [pourcentageNonConfient, pourcentageSemiConfient, pourcentageConfient, pourcentageNonConfientIfSame,
            pourcentageSemiConfientIfSame, pourcentageConfientIfSame, sucessRate]

def findCorrelationWithDonations(partyMember, finalVotes):

    totalCount = 0
    totalGaveToSameThanVote = 0
    totalGaveToDifferentThanVote = 0

    for i in range(0, len(partyMember)):
        if not pd.isna(partyMember.iloc[i]['cps19_fed_member']) and not pd.isna(finalVotes.iloc[i]['finalVote']):
            #print(partyMember.iloc[i]['cps19_fed_member'])
            totalCount += 1

        if not pd.isna(partyMember.iloc[i]['cps19_fed_member']) and not pd.isna(finalVotes.iloc[i]['finalVote']) \
                and partyMember.iloc[i]['cps19_fed_member'] == finalVotes.iloc[i]['finalVote']:
            #print( "Gave to: {gaveParty}  Final Vote: {finalVote}".format(
            #gaveParty=partyMember.iloc[i]['cps19_fed_member'], finalVote=finalVotes.iloc[i]['finalVote']))
            totalGaveToSameThanVote += 1

        if not pd.isna(partyMember.iloc[i]['cps19_fed_member']) and not pd.isna(finalVotes.iloc[i]['finalVote']) \
                and partyMember.iloc[i]['cps19_fed_member'] != finalVotes.iloc[i]['finalVote']:
            #print( "Gave to: {gaveParty}  Final Vote: {finalVote}".format(
            #gaveParty=partyMember.iloc[i]['cps19_fed_member'], finalVote=finalVotes.iloc[i]['finalVote']))
            totalGaveToDifferentThanVote += 1

    return [totalCount, totalGaveToSameThanVote, totalGaveToDifferentThanVote]


def findTrudeauSatisfactionCorrelation(currentTrudeauSatisfaction, finalVotes):
    verySatisfiedCount = 0
    fairlySatisfiedCount = 0
    notVerySatisfiedCount = 0
    notSatisfiedCount = 0
    dontKnowCount = 0

    verySatisfiedAndVotedForLiberalsCount = 0
    fairlySatisfiedAndVotedForLiberalsCount = 0
    notVerySatisfiedAndVotedForLiberalsCount = 0
    notSatisfiedAndVotedForLiberalsCount = 0
    dontKnowAndVotedForLiberalsCount = 0
    for i in range(0, len(currentTrudeauSatisfaction)):

        if  currentTrudeauSatisfaction.iloc[i]['cps19_fed_gov_sat'] == "Very satisfied":
            verySatisfiedCount +=1
            if  finalVotes.iloc[i]['finalVote'] == "Liberal Party":
                verySatisfiedAndVotedForLiberalsCount +=1
                #print("Sattisfcation: {satisfaction}  Final Vote: {finalVote}".format(
                    #satisfaction=currentTrudeauSatisfaction.iloc[i]['cps19_fed_gov_sat'], finalVote=finalVotes.iloc[i]['finalVote']))

        if  currentTrudeauSatisfaction.iloc[i]['cps19_fed_gov_sat'] == "Fairly satisfied":
            fairlySatisfiedCount +=1
            if  finalVotes.iloc[i]['finalVote'] == "Liberal Party":
                fairlySatisfiedAndVotedForLiberalsCount +=1

        if  currentTrudeauSatisfaction.iloc[i]['cps19_fed_gov_sat'] == "Not very satisfied":
            notVerySatisfiedCount +=1
            if  finalVotes.iloc[i]['finalVote'] == "Liberal Party":
                notVerySatisfiedAndVotedForLiberalsCount +=1

        if  currentTrudeauSatisfaction.iloc[i]['cps19_fed_gov_sat'] == "Not at all satisfied":
            notSatisfiedCount +=1
            if  finalVotes.iloc[i]['finalVote'] == "Liberal Party":
                notSatisfiedAndVotedForLiberalsCount +=1

        if  currentTrudeauSatisfaction.iloc[i]['cps19_fed_gov_sat'] == "Don't know/ Prefer not to answer":
            dontKnowCount +=1
            if  finalVotes.iloc[i]['finalVote'] == "Liberal Party":
                dontKnowAndVotedForLiberalsCount +=1


    print("Very Satisfied: {verySatisfied}  Voted For Liberals: {verySatisfiedAndVotedForLiberals}".format(
    verySatisfied=verySatisfiedCount, verySatisfiedAndVotedForLiberals=verySatisfiedAndVotedForLiberalsCount))

    print("Fairly Satisfied: {fairlySatisfied}  Voted For Liberals: {fairlySatisfiedAndVotedForLiberals}".format(
    fairlySatisfied=fairlySatisfiedCount, fairlySatisfiedAndVotedForLiberals=fairlySatisfiedAndVotedForLiberalsCount))

    print("Not very Satisfied: {notVerySatisfied}  Voted For Liberals: {notVerySatisfiedAndVotedForLiberals}".format(
    notVerySatisfied=notVerySatisfiedCount, notVerySatisfiedAndVotedForLiberals=notVerySatisfiedAndVotedForLiberalsCount))

    print("Not at all Satisfied: {notSatisfied}  Voted For Liberals: {notSatisfiedAndVotedForLiberals}".format(
    notSatisfied=notSatisfiedCount, notSatisfiedAndVotedForLiberals=notSatisfiedAndVotedForLiberalsCount))

    print("Dont know: {dontKnow}  Voted For Liberals: {dontKnowAndVotedForLiberals}".format(
    dontKnow=dontKnowCount, dontKnowAndVotedForLiberals=dontKnowAndVotedForLiberalsCount))

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
        print("NOUVELLE PERSONNE!")
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

    QUESTIONS_TO_USE = ['cps19_outcome_most', 'cps19_province', 'cps19_imp_iss_party', 'cps19_fed_id', 'cps19_fed_member',
                        'cps19_fed_gov_sat']


    onlyDesiredAnswers = data.loc[:, ['cps19_outcome_most', 'cps19_province', 'cps19_imp_iss_party', 'cps19_fed_id', 'cps19_fed_member', 'cps19_fed_gov_sat']]
    persons = onlyDesiredAnswers.iloc[[1, 3, 5, 7, 9, 11]]
    allResults = getFullResults(data, persons)
    combinedResults = getCombinedResults(allResults)
    finalVotePredictions = getFinalPredictions(combinedResults)
    print(finalVotePredictions)

    """
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

    """
    QUESTIONS_TO_USE = ['cps19_outcome_most', 'cps19_province', 'cps19_imp_iss_party', ' cps19_fed_id', 'cps19_fed_member',
                        'cps19_fed_gov_sat', 'pes19_provvote', 'cps19_prov_member', 'cps19_party_rating_24',
                        'cps19_lead_rating_24', 'cps19_cand_rating_24', 'pid_party_fr', 'pid_en', 'cps19_cand_rating_26',
                        'cps19_party_rating_25', 'pes19_votechoice2019']
    """



    #Getting the right rows without NaN to get correct probabilities
    #vote = data["finalVote"][~data['cps19_outcome_most'].isna()][~data['finalVote'].isna()]
    #desiredOutcome = data['cps19_outcome_most'][~data['cps19_outcome_most'].isna()][~data['finalVote'].isna()]
    """
    vote = data["finalVote"][~data['finalVote'].isna()]
    desiredOutcome = data['cps19_outcome_most'][~data['finalVote'].isna()]
    biggestIssue = data['cps19_imp_iss_party'][~data['finalVote'].isna()]

    #Encoding the string data into numerical values
    voteEncoded = le.fit_transform(vote)
    le_vote_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    desiredOutcomeEncoded = le.fit_transform(desiredOutcome)
    le_outcome_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    biggestIssueEncoded = le.fit_transform(vote)
    le_issue_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    features = zip(desiredOutcomeEncoded, biggestIssueEncoded)

    #Getting our Test and Training subsets
    X_train, X_test, y_train, y_test = train_test_split(features, voteEncoded, test_size=0.20, random_state=1)
    """


    """
    #Why is Gaussian much better than Multinomial and Bernouilli ????
    clf = GaussianNB()
    clf.fit(X_train.reshape(-1, 1), y_train)
    y_pred = clf.predict(X_test.reshape(-1, 1))
    successRate = accuracy_score(y_true=y_test, y_pred = y_pred)
    """

    #Print the predicted combinedResults
    """
    for y in y_pred:
        for vote, voteMappping in le_vote_mapping.items():
            if y == voteMappping:
                print(vote)
    """
    #print(successRate)
    #print(le_vote_mapping)
    #print(le_outcome_mapping)


if __name__ == '__main__':
    main()


"""
    desiredOutcome = data[['Unnamed: 0', 'cps19_outcome_most']]
    provinceAnswers = data[['Unnamed: 0', 'cps19_province']]
    bestAdressesIssue = data[['Unnamed: 0', 'cps19_imp_iss_party']]
    affinitePolitique = data[['Unnamed: 0', 'cps19_fed_id']]
    confidenceAffinete = data[['Unnamed: 0', "cps19_fed_id_str"]]
    partyMember = data[['Unnamed: 0', "cps19_fed_member"]]
    currentTrudeauSatisfaction = data[['Unnamed: 0', "cps19_fed_gov_sat"]]


    vote = np.where(pd.isna(data["cps19_votechoice"]), data["cps19_votechoice_pr"], data["cps19_votechoice"])
    vote = np.where(pd.isna(vote), data["cps19_vote_unlikely"], vote)
    vote = np.where(pd.isna(vote), data["cps19_vote_unlike_pr"], vote)
    vote = np.where(pd.isna(vote), data["cps19_v_advance"], vote)
    vote = np.where(pd.isna(vote), data["cps19_vote_lean"], vote)
    data["finalVote"] = vote
    finalVotes = data[['Unnamed: 0', 'finalVote']]


    #data[(data["pes19_province"] != "Quebec") & (finalVotes["vote"] == "Bloc Qu<e9>b<e9>cois")].count()[0]

    #print(finalVotes['finalVote'])
"""