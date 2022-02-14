import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


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
            count +=1
        actualVotes['finalVote'][i] = realVote
    return actualVotes

def findCorrelationWithQuebecProvince(provinceAnswers, finalVotes):
    provinceAnsweredCount = 0
    quebecResidentTotalCount = 0
    quebecResidentVotedForBlocCount = 0
    blocVoterNotQuebecResidentCount =0
    for i in range(0, len(provinceAnswers)):
        #print(finalVotes.iloc[i]['finalVote'])
        if not pd.isna(provinceAnswers.iloc[i]['pes19_province']):
            provinceAnsweredCount +=1

        if provinceAnswers.iloc[i]['pes19_province'] == 'Quebec':
            quebecResidentTotalCount +=1

        if provinceAnswers.iloc[i]['pes19_province'] == 'Quebec' and finalVotes.iloc[i]['finalVote'] == "Bloc Qu<e9>b<e9>cois":
            quebecResidentVotedForBlocCount +=1

        if provinceAnswers.iloc[i]['pes19_province'] != 'Quebec' and finalVotes.iloc[i]['finalVote'] == "Bloc Qu<e9>b<e9>cois":
            if not pd.isna(provinceAnswers.iloc[i]['pes19_province']):
                #print( provinceAnswers.iloc[i]['pes19_province'] , finalVotes.iloc[i]['finalVote'])
                blocVoterNotQuebecResidentCount +=1

    return [provinceAnsweredCount, quebecResidentTotalCount, quebecResidentVotedForBlocCount, blocVoterNotQuebecResidentCount]

def findCorrelationForBiggestIssue(bestAdressesIssue, finalVotes):
    bestAdressesPartyTotalCount = 0
    bestAdressesIssueSameForVoteCount = 0
    bestAdressesIssueDifferentForVoteCount = 0
    for i in range(0, len(bestAdressesIssue)):
        if not pd.isna(bestAdressesIssue.iloc[i]['cps19_imp_iss_party']) and not pd.isna(finalVotes.iloc[i]['finalVote'])\
                and finalVotes.iloc[i]['finalVote'] != "Don't know/ Prefer not to answer"\
                and bestAdressesIssue.iloc[i]['cps19_imp_iss_party'] != "Don't know/ Prefer not to answer":
            bestAdressesPartyTotalCount += 1

        if bestAdressesIssue.iloc[i]['cps19_imp_iss_party'] == finalVotes.iloc[i]['finalVote']\
                and (not pd.isna(bestAdressesIssue.iloc[i]['cps19_imp_iss_party']) and not pd.isna(finalVotes.iloc[i]['finalVote'])
        and finalVotes.iloc[i]['finalVote'] != "Don't know/ Prefer not to answer"
        and bestAdressesIssue.iloc[i]['cps19_imp_iss_party'] != "Don't know/ Prefer not to answer"):
                #print( "Best adresses: {bestAdresses}  Final Vote: {finalVote}".format(
                        #bestAdresses=bestAdressesIssue.iloc[i]['cps19_imp_iss_party'], finalVote=finalVotes.iloc[i]['finalVote']))
                bestAdressesIssueSameForVoteCount +=1

        if bestAdressesIssue.iloc[i]['cps19_imp_iss_party'] != finalVotes.iloc[i]['finalVote']\
                and (not pd.isna(bestAdressesIssue.iloc[i]['cps19_imp_iss_party']) and not pd.isna(finalVotes.iloc[i]['finalVote'])
        and finalVotes.iloc[i]['finalVote'] != "Don't know/ Prefer not to answer"
        and bestAdressesIssue.iloc[i]['cps19_imp_iss_party'] != "Don't know/ Prefer not to answer"):

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
                and finalVotes.iloc[i]['finalVote'] != "Liberal Party"\
                and finalVotes.iloc[i]['finalVote'] != "Don't know/ Prefer not to answer":
            desiredOutcomeDifferentThenVoteLiberals +=1

        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "Conservative majority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "Conservative minority")\
                and finalVotes.iloc[i]['finalVote'] == "Conservative Party":
            #print( "Best adresses: {bestOutcome}  Final Vote: {finalVote}".format(
            #bestOutcome=desiredOutcome.iloc[i]['cps19_outcome_most'], finalVote=finalVotes.iloc[i]['finalVote']))
            desiredOutcomeSameThenVoteConservative +=1


        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "Conservative majority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "Conservative minority")\
                and finalVotes.iloc[i]['finalVote'] != "Conservative Party"\
                and finalVotes.iloc[i]['finalVote'] != "Don't know/ Prefer not to answer":
            desiredOutcomeDifferentThenVoteConservative +=1

        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "NDP minority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "NDP majority")\
                and finalVotes.iloc[i]['finalVote'] == "ndp":
            #print( "Best adresses: {bestOutcome}  Final Vote: {finalVote}".format(
            #bestOutcome=desiredOutcome.iloc[i]['cps19_outcome_most'], finalVote=finalVotes.iloc[i]['finalVote']))
            desiredOutcomeSameThenVoteNDP +=1


        if (desiredOutcome.iloc[i]['cps19_outcome_most'] == "NDP minority" or desiredOutcome.iloc[i]['cps19_outcome_most'] == "NDP majority")\
                and finalVotes.iloc[i]['finalVote'] != "ndp"\
                and finalVotes.iloc[i]['finalVote'] != "Don't know/ Prefer not to answer":
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
                    and affinitePolitique.iloc[i]['cps19_fed_id'] != "Another party (please specify)" \
                    and finalVotes.iloc[i]['finalVote'] != "None of these" \
                    and finalVotes.iloc[i]['finalVote'] != "Don't know/ Prefer not to answer" \
                    and finalVotes.iloc[i]['finalVote'] != "Another party (please specify)" :
                nbrDifferent += 1

                if (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Not very strongly"):
                    notConfident += 1
                elif (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Fairly strongly"):
                    semiConfident += 1
                elif (confidenceAffinete.iloc[i]['cps19_fed_id_str'] == "Very strongly"):
                    confident += 1

                print(
                    "Affinite = {affinitePolitique} , vrai vote =  {voteChoice}, Confiance = {confiance}".format(
                        affinitePolitique=affinitePolitique.iloc[i]['cps19_fed_id'], voteChoice=finalVotes.iloc[i]['finalVote'], confiance=confidenceAffinete.iloc[i]['cps19_fed_id_str']))

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
            print( "Gave to: {gaveParty}  Final Vote: {finalVote}".format(
            gaveParty=partyMember.iloc[i]['cps19_fed_member'], finalVote=finalVotes.iloc[i]['finalVote']))
            totalGaveToDifferentThanVote += 1

    return [totalCount, totalGaveToSameThanVote, totalGaveToDifferentThanVote]



def main():
    desiredOutcome = data[['Unnamed: 0', 'cps19_outcome_most']]
    provinceAnswers = data[['Unnamed: 0', 'pes19_province']]
    bestAdressesIssue = data[['Unnamed: 0', 'cps19_imp_iss_party']]
    affinitePolitique = data[['Unnamed: 0', 'cps19_fed_id']]
    confidenceAffinete = data[['Unnamed: 0', "cps19_fed_id_str"]]
    partyMember = data[['Unnamed: 0', "cps19_fed_member"]]
    currentTrudeauSatisfaction = data[['Unnamed: 0', "cps19_fed_gov_sat"]]

    actualVotes = data[['Unnamed: 0', "cps19_votechoice", "cps19_votechoice_pr", "cps19_vote_unlikely", "cps19_vote_unlike_pr", "cps19_v_advance", 'finalVote']]

    finalVotes = findFinalVotes(actualVotes)[['Unnamed: 0', 'finalVote']]

    #quebecVotersData = findCorrelationWithQuebecProvince(provinceAnswers, finalVotes)
    #biggestIssueData = findCorrelationForBiggestIssue(bestAdressesIssue, finalVotes)
    #mostWantedOutcomeData = findCorrelationMostWantedOutcome(desiredOutcome, finalVotes)
    #affiliationPhiloDaa = findCorrelationAffiliationPgilosophique(affinitePolitique, confidenceAffinete, actualVotes)
    #gaveMoneyToParty = findCorrelationWithDonations(partyMember, finalVotes)

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


if __name__ == '__main__':
    main()

"""
for i in desiredOutcome['cps19_outcome_most']:
    if not pd.isna(i):
        print(i)
"""


#print(desiredOutcome)