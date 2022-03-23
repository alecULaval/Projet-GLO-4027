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


def main():
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
    nanCount = 0
    for i in range(0, len(finalVotes)):
        #print(finalVotes.iloc[i]['Unnamed: 0'], finalVotes.iloc[i]['finalVote'])
        if pd.isna(finalVotes.iloc[i]['finalVote']):
            nanCount +=1
    print(nanCount)
    """

    #quebecVotersData = findCorrelationWithQuebecProvince(provinceAnswers, finalVotes)
    #print(quebecVotersData)
    #biggestIssueData = findCorrelationForBiggestIssue(bestAdressesIssue, finalVotes)
    #print(biggestIssueData)
    #mostWantedOutcomeData = findCorrelationMostWantedOutcome(desiredOutcome, finalVotes)
    #print(mostWantedOutcomeData)
    #affiliationPhiloData = findCorrelationAffiliationPgilosophique(affinitePolitique, confidenceAffinete, finalVotes)
    #print(affiliationPhiloData)
    #gaveMoneyToPartyData = findCorrelationWithDonations(partyMember, finalVotes)
    #print(gaveMoneyToPartyData)
    #trudeauSatisfactionData = findTrudeauSatisfactionCorrelation(currentTrudeauSatisfaction, finalVotes)

    le = LabelEncoder()

    #Getting the right rows without NaN to get correct probabilities
    vote = data["finalVote"][~data['cps19_outcome_most'].isna()][~data['finalVote'].isna()]
    desiredOutcome = data['cps19_outcome_most'][~data['cps19_outcome_most'].isna()][~data['finalVote'].isna()]

    #Encoding the string data into numerical values
    desiredOutcomeEncoded = le.fit_transform(desiredOutcome)
    le_outcome_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    voteEncoded = le.fit_transform(vote)
    le_vote_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    #Getting our Test and Training subsets
    X_train, X_test, y_train, y_test = train_test_split(desiredOutcomeEncoded, voteEncoded, test_size=0.20, random_state=1)

    #Why is Gaussian much better than Multinomial and Bernouilli ????
    clf = GaussianNB()
    clf.fit(X_train.reshape(-1, 1), y_train)
    y_pred = clf.predict(X_test.reshape(-1, 1))
    successRate = accuracy_score(y_true=y_test, y_pred = y_pred)

    #Print the predicted results
    for y in y_pred:
        for vote, voteMappping in le_vote_mapping.items():
            if y == voteMappping:
                print(vote)
    print(successRate)
    print(le_vote_mapping)
    print(le_outcome_mapping)


if __name__ == '__main__':
    main()

"""
for i in desiredOutcome['cps19_outcome_most']:
    if not pd.isna(i):
        print(i)
"""


#print(desiredOutcome)