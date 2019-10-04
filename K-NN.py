#K-Nearest Neighbor Implementation Project
#Alexander Alvarez
#Matt Wintersteen
#Kyle Webster
import math

#generalized minkowski distance, where p is either input integer or string 'inf'
def minkowskiDistance(v1,v2,p):
    if type(p)==str:
        maxDistance = 0
        for x in range(len(v1)):
            maxDistance = max(maxDistance, abs(v1[x]-v2[x]))
        return maxDistance
    else:
        distance = 0
        # assume: v1 and v2 are equal length
        for x in range(len(v1)-1):
            distance += pow((abs(v1[x]-v2[x])),p)
        return pow(distance, 1.0/p)

#class for storing the data sets
class dataset:
    total_set = []
    training_set = [[]]
    test_set = [[]]

    def __init__(self,file_name):
        total_set = []
        #open input and output files
        with open(file_name) as readIn:
            #iterate over each line in input file
            for line in readIn:
                features = line.split(",")
                total_set.append(features)
        self.total_set = total_set

    def k_split(k):
        test_size = len(total_set.length)
        training_set = [[]]
        test_set = [[]]

        for i in range(0, k*test_size):
            training_set.append(total_set[i])
        for i in range(k*test_size, (k+1)*test_size):
            test_set.append(total_set[i])
        for i in range((k+1)*test_size, len(total_set)):
            training_set.append(total_set[i])

#class containing methods for preprocessing the datasets
class pre_processing:
    def remove_headers(data):
        del data.total_set[0]

#class containing methods implementing the K-NN algorithms
class k_nearest_neighbor:
    def __init__(self):
        print("init knn")
    def knn(trainingSet, t, k):
        distances = []
        for x in range(len(trainingSet)):
            dist = minkowskiDistance(t, trainingSet[x], 2)
            distances.append((trainingSet[x], dist))
        # Sort by the second value in sub list
        distances.sort(key = lambda x: x[1])
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors
    
    def getClass(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(),key = lambda x: x[1],reverse=True)
        return sortedVotes[0][0]
    
    def getAccuracy(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] is predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) *100.0
    
    def k_nearest(data, k):
        dataset.k_split(k)

#class for driving the program
class main:
    
    abalone = dataset("data/abalone.data")
    car = dataset("data/car.data")
    forest_fires = dataset("data/forestfires.csv")
    machine = dataset("data/machine.data")
    segmentation = dataset("data/segmentation.data")
    wine_red = dataset("data/winequality-red.csv")
    wine_white = dataset("data/winequality-white.csv")


    trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b'],[2, 3, 2, 'a'], [4, 4, 5, 'b']]
    testSet = [[5, 5, 5, 'b'],[2, 2, 1, 'a']]
    k = 1
    predictions = []
    k_nearest_neighbor()
    for x in range(len(testSet)):
        neighbors = k_nearest_neighbor.knn(trainSet, testSet[x], k)
        result = k_nearest_neighbor.getClass(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = k_nearest_neighbor.getAccuracy(testSet,predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


driver = main()
#main.run()
