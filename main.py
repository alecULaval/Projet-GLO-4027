import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data = pd.read_csv("donnees.csv")
columns = data.columns

nbrGivenInAnswer = 0

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

print(len(canadians1979))

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

"""
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