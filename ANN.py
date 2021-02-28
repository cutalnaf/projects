import numpy
from math import sqrt
import sklearn
from sklearn.metrics import mean_squared_error

def sigmoid(inpt):
    return 1.0 / (1.0 + numpy.exp(-1 * inpt))

def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result

def predict_outputs(weights_mat, data_inputs, data_outputs, activation="relu"):
    RMSE = list()
    predictions = numpy.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights in weights_mat:
            r1 = numpy.dot(a=r1, b=curr_weights)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    rmse = sqrt(sklearn.metrics.mean_squared_error(data_outputs, predictions))
    accuracy = (correct_predictions / data_outputs.size) * 100
    print('accuracy:', accuracy)
    print('rmse:', rmse)
    return accuracy, predictions, rmse

def fitness(weights_mat, data_inputs, data_outputs, activation="relu"):
    accuracy = numpy.empty(shape=(weights_mat.shape[0]))
    for sol_idx in range(weights_mat.shape[0]):
        curr_sol_mat = weights_mat[sol_idx, :]
        accuracy[sol_idx], rmse, _ = predict_outputs(curr_sol_mat, data_inputs, data_outputs, activation=activation)
        print('what is accuracy[sol_idx]', accuracy[sol_idx])
    return accuracy