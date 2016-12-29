#Copyright of Yakup Gorur
import numpy as np
import pandas as pd

alphas = [0.00001,0.0001, 0.001, 0.01, 1, 10, 100]

def Error(output,desired_output):


    output=np.negative(output)


    error=np.add(desired_output,output)

    error = np.abs(error)

    error=np.square(error)
    error=np.mean(error)
    error= np.sqrt(error)
    return error



#input file
traindata= pd.read_csv('/Users/yakup/Downloads/Predicting a Biological Response/train.csv') #Open file
traindata=traindata.iloc[np.random.permutation(len(traindata))] #mix the order



traindatai=traindata
traindatao=traindatai.as_matrix(columns=traindatai.columns[:1])
traindatai = traindatai.drop('Activity', 1) # drop the outputs



# its not sigmoid**** tanh
def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))
#def sigmoid(x):
#    output = 1 / (1 + np.exp(-x))
#    return output


# ****tanh
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


X = traindatai.values

y = traindatao
f = open('Error.txt', 'w')
for alpha in alphas:
    print "\nTraining  With Alpha:" + str(alpha)
    np.random.seed(1)

    #randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((1776, 11)) - 1
    synapse_1 = 2 * np.random.random((11, 4)) - 1
    synapse_2 = 2 * np.random.random((4, 2)) - 1
    synapse_3 = 2 * np.random.random((2, 1)) - 1

    for j in xrange(20000):


        layer_0=X

        layer_1=sigmoid(np.dot(layer_0,synapse_0))

        layer_2=sigmoid(np.dot(layer_1,synapse_1))

        layer_3=sigmoid(np.dot(layer_2,synapse_2))

        layer_4=sigmoid(np.dot(layer_3,synapse_3))


        layer_4_error= layer_4 - y


        if(j%1000) == 999:
            print "Error After:" + str(j)+ " iterations:" + str(np.mean(np.abs(layer_4_error)))


        layer_4_delta = layer_4_error * sigmoid_output_to_derivative(layer_4)



        layer_3_error = layer_4_delta.dot(synapse_3.T)



        layer_3_delta = layer_3_error * sigmoid_output_to_derivative(layer_3)


        layer_2_error = layer_3_delta.dot(synapse_2.T)


        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)


        layer_1_error = layer_2_delta.dot(synapse_1.T)


        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)





        synapse_3 -= alpha * (layer_3.T.dot(layer_4_delta))
        synapse_2 -= alpha * (layer_2.T.dot(layer_3_delta))
        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))


    #this is just testing
    #f = open('Weights.txt', 'wr')
    #f.write(str(synapse_3))
    #f.write('\n')
    #f.write(str(synapse_2))
    #f.write('\n')
    #f.write(str(synapse_1))
    #f.write('\n')
    #f.write(str(synapse_0))
    #f.close()

    #Input file
    test_data = pd.read_csv('/Users/yakup/Downloads/Predicting a Biological Response/test.csv')  # Open file
    x = test_data.values
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    layer_3 = sigmoid(np.dot(layer_2, synapse_2))
    layer_4 = sigmoid(np.dot(layer_3, synapse_3))


    outputfile= 'outputfile_' + str(alpha) + '.csv'
    df = pd.DataFrame(layer_4, columns=['PredictedProbability'],index=np.arange(1,2502))
    df.to_csv(outputfile, sep=',', float_format='%.6f', index_label=['MoleculeId'])

    #Input file
    test = pd.read_csv('/Users/yakup/Downloads/Predicting a Biological Response/svm_benchmark.csv')  # Open file
    testo= test.as_matrix(columns=test.columns[1:])


    error=Error(layer_4,testo)


    f.write('alpha='+str(alpha)+'  iterate='+str(j+1) + '  error: ' + str(np.mean(np.abs(layer_4 - testo)))  +'   standart  error='+str(error)  + '\n')

f.close()















