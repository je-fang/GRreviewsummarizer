import cleaning
import collectreviews
import rbm2

import numpy as np

def create_rfile():
    r = collectreviews.Reviews()
    r.output_reviews()
    return r.file_name

def generate_top(file_name):
    (sent_list, features_matrix) = cleaning.create_features(file_name + ".txt")

    r = rbm2.RBM(num_visible = 6, num_hidden = 6)
    training_data = features_matrix
    r.train(training_data, max_epochs = 50)

    temp = np.dot(features_matrix, np.transpose(r.weights[1:, 1:]))

    score_list = [sum(temp[x]) for x in range(len(sent_list))]

    sorted_index = np.argsort(score_list)
    print(sorted_index)

    for i in range(5):
        print(sent_list[sorted_index[-1-i]])

file_name = create_rfile()
generate_top(file_name)
