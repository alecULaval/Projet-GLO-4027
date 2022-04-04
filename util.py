
exempleFile = "exemple.txt"
resultFile = "results.txt"

def readResultFile(file=exempleFile):
    with open(file) as f:
        lines = f.readlines()
        rows = [line.split("\t")[0] for line in lines]
        return rows

def writeResultFile(data, classifier, file=resultFile):
    predictions = classifier.predict(data)
    indices = data.index

    with open(file, 'w') as f:
        for prediction, indice in zip(predictions, indices):
            f.write("{}\t{}\n".format(indice, prediction))