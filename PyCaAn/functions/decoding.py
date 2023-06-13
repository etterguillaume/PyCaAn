import numpy as np
from sklearn.neighbors import KNeighborsRegressor as knn_reg
from sklearn.neighbors import KNeighborsClassifier as knn_class
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import median_absolute_error as MAE
#from sklearn.metrics import f1_score
from tqdm import tqdm

def decode_neural_data(var2predict, neural_data, params, trainingFrames, testingFrames): #TODO optimize speed
    np.random.seed(params['seed'])
    decoder = BayesianRidge().fit(neural_data[trainingFrames], var2predict[trainingFrames])
    prediction = decoder.predict(neural_data[testingFrames])
    decoding_score = decoder.score(neural_data[testingFrames], var2predict[testingFrames])
    decoding_error = MAE(var2predict[testingFrames],prediction)

    shuffled_score = np.zeros(params['num_surrogates'])
    shuffled_error = np.zeros(params['num_surrogates'])
    for shuffle_i in tqdm(range(params['num_surrogates'])):
        idx = np.random.randint(len(neural_data))
        shuffled_var = np.concatenate((var2predict[idx:], var2predict[:idx]))
        decoder = knn_reg().fit(neural_data[trainingFrames], shuffled_var[trainingFrames])
        prediction = decoder.predict(neural_data[testingFrames])
        shuffled_score[shuffle_i] = decoder.score(neural_data[testingFrames],shuffled_var[testingFrames])
        shuffled_error[shuffle_i] = MAE(var2predict[testingFrames],prediction)

    decoding_zscore = (decoding_score-np.mean(shuffled_score))/np.std(shuffled_score)
    decoding_pvalue = np.sum(shuffled_score>decoding_score)/params['num_surrogates']
    shuffled_error = np.nanmean(shuffled_error)
    return decoding_score, decoding_zscore, decoding_pvalue, decoding_error, shuffled_error

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

def RWI_decoding(data, params, embedding):
    np.random.seed(params['seed'])

    prediction_stats = np.zeros(len(params['num_k']))*np.nan
    
    # Build a new variable containing all internally generated signals
    internal_var = np.vstack((data['elapsed_time'],
           data['time2stop'],
           data['distance_travelled'],
           data['distance2stop'],
           data['velocity'])).T
    
    # Find optimal k
    for i, num_k  in enumerate(params['num_k']):
        external_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(embedding[data['trainingFrames']], data['position'][data['trainingFrames']])
        internal_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(embedding[data['trainingFrames']], internal_var[data['trainingFrames']])
        prediction_stats[i] = external_decoder.score(embedding['testingFrames'],data['position'][data['testingFrames']]) + internal_decoder.score(embedding['testingFrames'],internal_var[data['testingFrames']])

    optimal_k = np.argmax(prediction_stats)
    decoding_score = prediction_stats[optimal_k]

    external_decoder = knn_reg(metric='euclidean', n_neighbors=optimal_k).fit(embedding[data['trainingFrames']], data['position'][data['trainingFrames']])
    internal_decoder = knn_reg(metric='euclidean', n_neighbors=optimal_k).fit(embedding[data['trainingFrames']], internal_var[data['trainingFrames']])
    external_prediction = external_decoder.predict(embedding) # predict location
    internal_prediction = internal_decoder.predict(embedding) # predict internal signals (time, distance, speed)

    



    RWI = (external_info-internal_info)/(external_info+internal_info)

    return RWI, decoding_score