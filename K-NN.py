#K-Nearest Neighbor Implementation Project
#Alexander Alvarez
#Matt Wintersteen
#Kyle Webster
#Greg Martin

import math
import random
import time
from operator import add

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

#randomize data so that when we select training and test sets, we get a variety of each class
def randomizeData(data):
    randomSet = []
    copy = list(data)
    while len(randomSet) < len(data):
        index = random.randrange(len(copy))
        randomSet.append(copy.pop(index))
    return randomSet

#class for storing the data sets
class dataset:
    
    total_set = [] #total data set
    training_set = [] #list of training sets 
    test_set = [] #list of test sets (respective to training sets)

    def __init__(self,data):
        self.total_set = data
        self.kFoldCross(10)

    def getTotalSet(self):
        return self.total_set

    def getTrainingSet(self):
        return self.training_set

    def getTestSet(self):
        return self.test_set

    #k is number of training/test set pairs
    def kFoldCross(self, k):
        training_set = []
        test_set = []
        splitRatio = .9
        for i in range(k):
            testSize = int(len(self.total_set) - len(self.total_set) * splitRatio)
            index = i*testSize

            trainSet = list(self.total_set)
            testSet = []

            for j in range(testSize):
                testSet.append(trainSet.pop(index))

            training_set.append(trainSet)
            test_set.append(testSet)
        self.training_set = training_set
        self.test_set = test_set

#class containing methods for preprocessing the datasets
class pre_processing:
    data = []
    def __init__(self, file_name):
        
        data = []
        
        #open input and output files
        with open(file_name) as readIn:
            
            #iterate over each line in input file
            for line in readIn:
                if(file_name[:16] == "data/winequality"):
                    features = line.split(";")
                else:
                    features = line.split(",")
                data.append(features)

        #dataset-dependent operations
        if(file_name == "data/forestfires.csv" or file_name == "data/winequality-red.csv" or file_name ==  "data/winequality-white.csv"):
            data = self.removeHeaders(data, 1)
        elif(file_name == "data/segmentation.data"):
            data = self.removeHeaders(data, 5)

        #move class to rightmost column
        if(file_name == "data/segmentation.data" or file_name == "data/car.data"):
            data = self.moveColumn(data)

        #remove strings
        data = self.removeStrings(data)

        self.data = data
        
    def removeHeaders(self, data, rows):
        for i in range(rows):
            del data[0]

        print("Deleted Header Row")
        return data

    def moveColumn(self, data):
        for i in range(len(data)):
            l = len(data[i]) - 1 # for indexing the last element of the list
            temp = data[i][0]
            data[i][0] = data[i][l]
            data[i][l] = temp
        print("Moved first column to last column")
        return data

    def removeStrings(self, data):
        stringlist = []
        for i in range(len(data)):
            for j in range(len(data[i])-1):
                d = data[i][j].strip()
                
                try:
                    data[i][j] = float(d)
                except ValueError:
                    if(d not in stringlist):
                        stringlist.append(d)
                        d = len(stringlist)
                    else:
                        d = stringlist.index(d)
                    data[i][j] = float(d)
        if(len(stringlist) > 0):
            print("Removed Strings")
            print(stringlist)
        return data
    #Converts data into a Value Difference Matrix Probabilities for distance calculations
    def processClassification(self, inData, fileName):
        #Dictionary for probability conversions
        table = {}
        #Stores all classes for numberical conversions later
        classes = []

        #Generates and maps classes to nested dictinary, sorted by class, attribute column, and individual values
        for c in inData:
            if c[-1] not in classes:
                classes.append(c[-1])
            table.setdefault(classes.index(c[-1]), {})
            for idx, a in enumerate(c[:len(c)-1]):
                try:
                    table[classes.index(c[-1])][idx+1][a] += 1
                except:
                    table[classes.index(c[-1])].setdefault(idx+1, {})
                    table[classes.index(c[-1])][idx+1].setdefault(a,1)
                    table[classes.index(c[-1])][idx+1][a] +=1
        #creates probability table within dictionary
        print("Classification Probability Table")
        for key in table:
            for x in table[key]:
                total = 0
                for a in table[key][x]:
                    total += table[key][x].get(a)
                for a in table[key][x]:
                    table[key][x][a] /= float(total)
                print("Class:", key, "Attribute:", x, "Values:", table[key][x])
        #Uses the values in dictionary to convert the input data
        for i,c in enumerate(inData):
            for idx, a in enumerate(c[:len(c)-1]):
                try:
                    temp = classes.index(c[-1])
                    inData[i][0] = temp
                    inData[i][idx+1] = table[temp][idx+1][a]
                except:
                    inData[i][0] = c[-1]
                    inData[i][idx + 1] = table[c[-1]][idx + 1][int(a)]
        #pause for video
        #input("")
        return(inData)

    def getData(self):
        return self.data



#class containing methods implementing the K-NN algorithms
class k_nearest_neighbor:
    def __init__(self):
        print("init knn")

    @staticmethod
    def knn(trainingSets, t, k):
        
        distances = []

        #calculate distances for each training set
        for x in range(len(trainingSets)):
            dist = minkowskiDistance(t, trainingSets[x], 1)
            distances.append((trainingSets[x], dist))

        #find k nearest neighbors
        distances.sort(key = lambda x: x[1])
        neighbors = []

        for x in range(k):
            neighbors.append(distances[x][0])

        return neighbors

    #calculate class from neighbors using voting
    @staticmethod
    def getClass(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][len(neighbors[x])-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(),key = lambda x: x[1],reverse=True)
        return sortedVotes[0][len(sortedVotes[0])-1]
    #calculate mean from neighbors
    @staticmethod
    def getMean(neighbors):
        total = 0.0
        for x in range(len(neighbors)):
            response = neighbors[x][len(neighbors[x])-1]
            total += response

        avg = float(total)/len(neighbors)
        
        return avg

    #edit training sets using test sets
    def editSets(self, trainingSets, testSets, k):

        print("Editing Sets...")
        editedSets = []
        
        for x in range(len(trainingSets)):

            trainingSet = trainingSets[x]
            testSet = testSets[x]
            editedSet = trainingSet[:]
            change = 1.0
            newAccuracy = 0

            #repeat
            while (True):
                
                tagged = []

                #tag points for removal
                for i in range(len(editedSet)):
                    point = editedSet[i]
                    if (self.getClass(self.knn(trainingSet, point, k)) != point[len(point)-1]):
                        tagged.append(point)
                #True is for the method, we want to use classification
                oldAccuracy = self.getClassificationPerformance(True,editedSet, testSet, k)

                #remove points
                for tag in tagged:
                    editedSet.remove(tag)

                newAccuracy = self.getClassificationPerformance(True,editedSet, testSet, k)

                change = abs(newAccuracy - oldAccuracy)

                #until there is no change
                if (change < 0.1): # threshold value
                    break

            editedSets.append(editedSet)
            num_removed = len(trainingSets[x])-len(editedSet)
            print("{} rows have been edited out".format(num_removed))

        return editedSets

    #edit training sets using test sets
    def condenseSets(self, trainingSets, testSets, k):

        print("Condensing Sets...")
        condensedSets = []
        
        for n in range(len(trainingSets)):

            trainingSet = trainingSets[n]
            condensedSetBefore = []
            condensedSetAfter = []
            while(True):
                for i in range(len(trainingSet)):
                    x = trainingSet[i]
                    condensedSetBefore = condensedSetAfter
                    if(condensedSetBefore == []):
                        condensedSetAfter.append(x)
                        condensedSetBefore = condensedSetAfter
                    else:
                        if(len(condensedSetAfter) < k):
                            neighbors = self.knn(condensedSetAfter, x, 1)
                        else:
                            neighbors = self.knn(condensedSetAfter, x, k)
                        if(neighbors[0][len(neighbors[0])-1] != x[len(x)-1]):
                            condensedSetAfter.append(x)
                            condensedSetBefore = condensedSetAfter
                if(condensedSetBefore == condensedSetAfter):
                    break
                
            condensedSets.append(condensedSetAfter)
            
        return condensedSets
    #Reducing dataset to centroids centered around the mean
    def kMeans(self, data, k):
        u = []
        change = 1
        for i in range(k):
            u.append(random.choice(data))
        while change > .01:
            centroids = {}
            for x in data:
                minDistance = None
                min = None
                for m in u:
                    dist = minkowskiDistance(x,m,2)
                    if minDistance == None:
                        minDistance = dist
                        min = m
                    elif dist < minDistance:
                        minDistance = dist
                        min = m
                a = u.index(min)
                try:
                    centroids[a].append(x)
                except:
                    centroids.setdefault(a, [])
                    centroids[a].append(x)
            for i in u:
                a = u.index(i)
                try:
                    temp = centroids[a]
                except:
                    del u[u.index(i)]
                total = temp[0]
                count = 1
                for j in temp[1:]:
                    total = list(map(add, total, j))
                    count += 1
                mean = [x / float(count) for x in total]
                oldU = u
                try:
                    u[u.index(i)] = mean
                except:
                    mean = mean
            comb = 0
            countC = 0
            for i in range(len(u)):
                comb += minkowskiDistance(u[i], oldU[i], 2)
                countC += 1
            change = comb/float(countC)
        print("Means \n",u)
        return u

    def kMedoids(self, data, k):

        return None

     #runs a single training/test set, returns accuracy
    def getClassificationPerformance(self, method, trainingSet, testSet, k):

        predictions = []
        for x in range(len(testSet)):
            neighbors = self.knn(trainingSet, testSet[x], k)
            if(method):
                result = float(self.getClass(neighbors))
            else:
                result = float(self.getMean(neighbors))
            predictions.append(result)
            #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][len(testSet[x])-1]))
        if(method):
            #for classification we just check if the prediction class = the test set
            return k_nearest_neighbor.getAccuracy(testSet,predictions)
        else:
            #for regression, I have decided to just use MSE (we can change this)
            return k_nearest_neighbor.getMSE(testSet,predictions)


    def getAccuracy(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if (testSet[x][len(testSet[x])-1] == predictions[x]):
                correct += 1

        return (correct/float(len(testSet))) *100.0
    
    def getMSE(testSet,predictions):
        total = 0
        size = len(testSet)
        for i in range(0,size):
            dif_squared = (predictions[i] - testSet[i][len(testSet[x])-1])**2
            total += dif_squared
        MSE = total/size
        return MSE


#class for driving the program
class main:
    files = ["data/segmentation.data",
             "data/car.data",
             "data/winequality-red.csv",
             "data/abalone.data",
             "data/machine.data",
             "data/forestfires.csv",
             "data/winequality-red.csv",
             "data/winequality-white.csv"]

    classification = ["data/segmentation.data",
                      "data/car.data",
                      "data/abalone.data"]
    
    regression = ["data/forestfires.csv",
                  "data/machine.data",
                  "data/winequality-red.csv",
                  "data/winequality-white.csv"]

    #classifies test sets using respective training sets, returns overall accuracy
    def run_knn(method, knn_instance, training_sets, test_sets, k):

        overall_accuracy = 0

        #caclulate accuracy of each training/test set pair
        for i in range(len(training_sets)):
            accuracy = knn_instance.getClassificationPerformance(method, training_sets[i], test_sets[i], k)
            overall_accuracy += accuracy

        overall_accuracy /= len(training_sets);

        print ("Accuracy: " + repr(overall_accuracy))

    knn_instance = k_nearest_neighbor()
    
    #for each classification data set
    for f in files:
        # method will be True for classification, false for regression
        method = True
        if(f in regression):
            method = False

        #import and process data set
        print("//////////\n{}\n//////////".format(f))
        p = pre_processing(f)
        inData = []
        #Categorical classification datasets converted
        if f in classification:
            inData = p.processClassification(p.getData(), f)
        else:
            inData = p.getData()
        randomizedData = randomizeData(inData)
        data = dataset(randomizedData)
        
        #get all training sets
        training_sets = data.getTrainingSet()
        test_sets = data.getTestSet()
        edited_sets = None
        condensed_sets = None
        if (method):
            edited_sets = knn_instance.editSets(training_sets, test_sets, 3)
            condensed_sets = knn_instance.condenseSets(training_sets, test_sets, 3)
        centroidsMean = []
        for j,i in enumerate(edited_sets):
            print("K-Means")
            centroidsMean.append(knn_instance.kMeans(training_sets[j], len(i)))

        #for each value of k, run algorithms
        for k in range(3,6):
            print("\n//////////\nk = " + repr(k) + "\n//////////")
            print("K-NN")
            run_knn(method, knn_instance, training_sets, test_sets, k)
            # We only run edited and condensed on classification datasets (method = True)
            if (method):
                print("Edited K-NN")
                run_knn(method, knn_instance, edited_sets, test_sets, k)
                print("Condensed K-NN")
                run_knn(method, knn_instance, condensed_sets, test_sets, k)
                print("K-Means Clustering")
                run_knn(method, knn_instance, centroidsMean, test_sets, k)

