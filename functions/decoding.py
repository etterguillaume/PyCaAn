import numpy as np
from sklearn.neighbors import KNeighborsRegressor as knn_reg
from sklearn.neighbors import KNeighborsClassifier as knn_class
from sklearn.metrics import median_absolute_error as MAE
#from sklearn.metrics import f1_score

def decode_embedding(var2predict,data, params, train_embedding, test_embedding):
    np.random.seed(params['seed'])
    prediction_stats = np.zeros(len(params['num_k']))*np.nan
    error_stats = np.zeros(len(params['num_k']))*np.nan
    for i, num_k  in enumerate(params['num_k']):
        if var2predict.dtype=='float': # Use kNN regressor
            decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(train_embedding, var2predict[data['trainingFrames']])
            prediction = decoder.predict(test_embedding)
            error_stats[i] = MAE(var2predict[data['testingFrames']],prediction)
        else: # Use kNN classifier
            decoder = knn_class(metric='euclidean', n_neighbors=num_k).fit(train_embedding, var2predict[data['trainingFrames']])
            #prediction = decoder.predict(test_embedding)
            #error_stats[i] = np.nan # could use f1 score? But then opposite of error

        prediction_stats[i] = decoder.score(test_embedding,var2predict[data['testingFrames']])

    optimal_k = np.argmax(prediction_stats)
    decoding_score = prediction_stats[optimal_k]
    decoding_error = error_stats[optimal_k]

    shuffled_score = np.zeros(params['num_surrogates'])
    shuffled_error = np.zeros(params['num_surrogates'])
    for shuffle_i in range(params['num_surrogates']):
        idx = np.random.randint(len(data['elapsed_time']))
        shuffled_var = np.concatenate((var2predict[idx:], var2predict[:idx]))
        prediction_stats = np.zeros(len(params['num_k']))*np.nan
        error_stats = np.zeros(len(params['num_k']))*np.nan
        for i, num_k  in enumerate(params['num_k']):
            if var2predict.dtype=='float': # Use kNN regressor
                decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(train_embedding, shuffled_var[data['trainingFrames']])
                prediction = decoder.predict(test_embedding)
                error_stats[i] = MAE(var2predict[data['testingFrames']],prediction)
            else: # Use kNN classifier
                decoder = knn_class(metric='euclidean', n_neighbors=num_k).fit(train_embedding, shuffled_var[data['trainingFrames']])
                #prediction = decoder.predict(test_embedding)
                #error_stats[i] = np.nan # could use f1 score? But then opposite of error

            prediction_stats[i] = decoder.score(test_embedding,shuffled_var[data['testingFrames']])

        shuffled_score[shuffle_i] = np.max(prediction_stats)
        optimal_k = np.argmax(prediction_stats)
        shuffled_score[shuffle_i] = prediction_stats[optimal_k]
        shuffled_error[shuffle_i] = error_stats[optimal_k]

    decoding_zscore = (decoding_score-np.mean(shuffled_score))/np.std(shuffled_score)
    decoding_pvalue = np.sum(shuffled_score>decoding_score)/params['num_surrogates']
    shuffled_error = np.nanmean(shuffled_error)
    return decoding_score, decoding_zscore, decoding_pvalue, decoding_error, shuffled_error