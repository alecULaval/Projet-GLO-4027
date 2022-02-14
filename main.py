import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data = pd.read_csv("CES19.csv")
columns = data.columns

nbrGivenInAnswer = 0


affinitePolitique = data['cps19_fed_id']
confidenceAffinete = data["cps19_fed_id_str"]
actualVote = data["cps19_votechoice"]
print(affinitePolitique)

nbrSame = 0
nbrDifferent =0
notConfident =0
semiConfident = 0
confident = 0

notConfidentIfSame =0
semiConfidentIfSame = 0
confidentIfSame = 0


for i in range(0, len(affinitePolitique)):

    if( affinitePolitique[i] == actualVote[i] or (affinitePolitique[i] == "Liberal" and actualVote[i] =="Liberal Party")
            or (affinitePolitique[i] == "Conservative" and actualVote[i] =="Conservative Party")
            or (affinitePolitique[i] == "Green" and actualVote[i] =="Green Party")):
        nbrSame +=1
        if(confidenceAffinete[i] == "Not very strongly"):
            notConfidentIfSame +=1
        elif(confidenceAffinete[i] == "Fairly strongly"):
            semiConfidentIfSame +=1
        elif (confidenceAffinete[i] == "Very strongly"):
            confidentIfSame +=1
    elif(not pd.isna(actualVote[i]) and actualVote[i]!= "Don't know/ Prefer not to answer"
        and affinitePolitique[i]!= "Don't know/ Prefer not to answer"
        and affinitePolitique[i]!= "None of these"
        and actualVote[i] != "Another party (please specify)"):
        nbrDifferent += 1

        if(confidenceAffinete[i] == "Not very strongly"):
            notConfident +=1
        elif(confidenceAffinete[i] == "Fairly strongly"):
            semiConfident +=1
        elif (confidenceAffinete[i] == "Very strongly"):
            confident +=1

        print(
            "Affinite = {affinitePolitique} , vrai vote =  {voteChoice}, Confiance = {confiance}".format(affinitePolitique=affinitePolitique[i], voteChoice=actualVote[i], confiance = confidenceAffinete[i]))

#Probleme 1: Beaucoup de gens se sont associé au Liberal mais ont voté NPD, vice-versa (confusion NPD-Lberaux). Le NPD fait chier, ils touchent un peu a tout.
#Probleme 2: Vote strategique

pourcentageNonConfient = notConfident / (notConfident +semiConfident +confident)*100
pourcentageSemiConfient = semiConfident / (notConfident +semiConfident +confident)*100
pourcentageConfient = confident / (notConfident +semiConfident +confident)*100

pourcentageNonConfientIfSame = notConfidentIfSame / (notConfidentIfSame +semiConfidentIfSame +confidentIfSame)*100
pourcentageSemiConfientIfSame = semiConfidentIfSame / (notConfidentIfSame +semiConfidentIfSame +confidentIfSame)*100
pourcentageConfientIfSame = confidentIfSame / (notConfidentIfSame +semiConfidentIfSame +confidentIfSame)*100


print("Parmi ceux qui avaient un vote different de leur affinite, {pourcentageNonConfient} % ne sont pas confiants, "
      "{pourcentageSemiConfient} % sont relativement confiants et {pourcentageConfient} % sont confiants".format(
    pourcentageNonConfient=pourcentageNonConfient, pourcentageSemiConfient=pourcentageSemiConfient, pourcentageConfient = pourcentageConfient))

print("Parmi ceux qui avaient LE MEME VOTE que celui de leur affinite, {pourcentageNonConfientIfSame} % ne sont pas confiants, "
      "{pourcentageSemiConfientIfSame} % sont relativement confiants et {pourcentageConfientIfSame} % sont confiants".format(
    pourcentageNonConfientIfSame=pourcentageNonConfientIfSame, pourcentageSemiConfientIfSame=pourcentageSemiConfientIfSame, pourcentageConfientIfSame = pourcentageConfientIfSame))

sucessRate = nbrSame /(nbrSame + nbrDifferent)*100
print(nbrSame + nbrDifferent)
print(sucessRate)
"""
missingValuesByColumns = []

#nbrQuestions = 0
for column in columns:
    amountOfMissingValues = 0
    #nbrQuestions += 1
    for datapoint in data[column]:
        if pd.isna(datapoint):
            amountOfMissingValues += 1

    missingValuesByColumns.append(amountOfMissingValues)


#print(len(missingValuesByColumns))

nbrCompleteAttribute = 0
for amount in missingValuesByColumns:
    if amount == 0:
        nbrCompleteAttribute += 1


datesDuSondage = data['cps19_StartDate']
dateElection = pd.Timestamp(2019, 10, 21)
dateLendemainElection = pd.Timestamp(2019, 10, 22)

nbrLateAnswer = 0
for date in datesDuSondage:
    if pd.Timestamp(date).date() == dateElection.date() or pd.Timestamp(date).date() == dateLendemainElection.date():
        nbrLateAnswer += 1

#print(nbrLateAnswer)

canadians1979 = (data[(data["cps19_yob"] == 1979) | (data["cps19_citizenship"] == "Permanent resident")]
         [["cps19_citizenship", "cps19_yob"]])

#print(len(canadians1979))

person101 = data.iloc[101][["cps19_citizenship", "cps19_yob"]]
#print(person101)


intelligentPoliticians = data.loc[:, 'cps19_lead_int_113':'cps19_lead_int_120']
nbrMoreThan3 = 0
for row in intelligentPoliticians.itertuples():
    nbrGivenInAnswer = 0
    for politicians in row:
        if pd.notnull(politicians):
            nbrGivenInAnswer += 1
    if nbrGivenInAnswer >= 3:
        nbrMoreThan3 += 1
#print("3 chefs ou plus intelligents: ", nbr_3_plus)


rhinosVoter = 0
dataOther = data[data["cps19_votechoice"] == "Another party (please specify)"]["cps19_votechoice_7_TEXT"]
for text in dataOther:
    if "rhino" in text or "Rhino" in text:
        rhinosVoter += 1
    else:
        print(text)
#print(rhinos)


#print(nbrCompleteAttribute)


votersNumber = data['Unnamed: 0']
voteChoices = data['cps19_votechoice']

bloc = 0
conservative = 0
liberal = 0
nan = 0
dontKnow = 0
ndp = 0

print(len(voteChoices))
print(len(votersNumber))

for i in range(len(voteChoices)):
    print("Le voteur numero {voterNumber} a vote pour {voteChoice}".format(voterNumber = votersNumber[i], voteChoice = voteChoices[i]))
"""