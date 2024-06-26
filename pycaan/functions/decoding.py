import numpy as np
from sklearn.linear_model import LinearRegression as lin_reg
from sklearn.neighbors import KNeighborsRegressor as knn_reg
from sklearn.neighbors import KNeighborsClassifier as knn_class
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import median_absolute_error as MAE
from math import dist
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

def bayesian_decode(tuning_curves, prior, marginal_likelihood, binary_data):
    '''
    Predict a behavioral variable with neural data using naive Bayes classifier.
    Uniform priors are recommanded unless strong priors are known.

    Returns posterior_probs and MAP (maximum a posteriori)
    '''
    numFrames, numNeurons = binary_data.shape
    numBins = tuning_curves.shape[1]

    posterior_probs = np.zeros((numFrames,numBins))*np.nan

    for frame_i in range(numFrames):
        bayesian_step_prob = np.empty((numNeurons,numBins))
        for cell_i in range(numNeurons):
            if binary_data[frame_i,cell_i]:
                active_tuning_curve = tuning_curves[cell_i]
                bayesian_step_prob[cell_i,:] = (active_tuning_curve*prior)/marginal_likelihood[cell_i]
            else:
                inactive_tuning_curve = 1-tuning_curves[cell_i]
                bayesian_step_prob[cell_i,:] = (inactive_tuning_curve*prior)/(1-marginal_likelihood[cell_i])
        
        posterior_probs[frame_i,:] = np.expm1(np.nansum(np.log1p(bayesian_step_prob),axis=0)) # This should be used instead of simple product to avoid numerical underflow
        posterior_probs[frame_i,:] = posterior_probs[frame_i,:]/np.nansum(posterior_probs[frame_i,:])    # Normalize into a probability distribution
        
        map = np.argmax(posterior_probs,axis=1)

    return (posterior_probs, map)

def temporal_bayesian_filter(posterior_probs, windowSize):
    '''
    Applies temporal filtering using window_size (in samples, should be even number).
    Uses a window step of 1, and automatically zero-pads start-end of input data.
    '''
    assert (windowSize % 2) == 0, "window_size should be an even number"

    smoothed_posteriors = np.zeros((posterior_probs.shape))*np.nan
    
    # Zero-padding
    posterior_probs = np.concatenate((
        np.zeros((int(windowSize/2),posterior_probs.shape[1])),
        posterior_probs,
        np.zeros((int(windowSize/2),posterior_probs.shape[1]))
        ))

    currentWindowIdx = np.arange(windowSize)

    for i in range(len(smoothed_posteriors)):
        bayesian_step_prob = posterior_probs[currentWindowIdx]
        smoothed_posteriors[i,:] = np.expm1(np.nansum(np.log1p(bayesian_step_prob),axis=0)) # This should be used instead of simple product to avoid numerical underflow
        smoothed_posteriors[i,:] = smoothed_posteriors[i,:]/np.nansum(smoothed_posteriors[i,:]) # Normalize into a probability distribution
        currentWindowIdx+=1 # Step forward

    return (smoothed_posteriors)

def decode_embedding(var2predict, data, params, train_embedding, test_embedding, isCircular):
    np.random.seed(params['seed'])
    prediction_stats = np.zeros(len(params['num_k']))*np.nan
    error_stats = np.zeros(len(params['num_k']))*np.nan
    for i, num_k  in enumerate(params['num_k']):
        if var2predict.dtype=='float': # Use kNN regressor
            decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(train_embedding, var2predict[data['trainingFrames']])
            test_prediction = decoder.predict(test_embedding)
            if isCircular:
                error_stats[i] = np.median(abs((var2predict[data['testingFrames']]-test_prediction+180)%360-180))
            else:
                error_stats[i] = MAE(var2predict[data['testingFrames']], test_prediction)
        else: # Use kNN classifier
            decoder = knn_class(metric='euclidean', n_neighbors=num_k).fit(train_embedding, var2predict[data['trainingFrames']])
            test_prediction = decoder.predict(test_embedding)
            #error_stats[i] = np.nan # could use f1 score? But then opposite of error

        prediction_stats[i] = decoder.score(test_embedding,var2predict[data['testingFrames']])

    optimal_k = np.argmax(prediction_stats)
    decoding_score = prediction_stats[optimal_k]
    decoding_error = error_stats[optimal_k]

    shuffled_score = np.zeros(params['num_surrogates'])
    shuffled_error = np.zeros(params['num_surrogates'])
    for shuffle_i in range(params['num_surrogates']):
        shuffled_var = var2predict[data['trainingFrames']]
        idx = np.random.randint(len(shuffled_var))
        shuffled_var = np.concatenate((shuffled_var[idx:], shuffled_var[:idx]))
        if var2predict.dtype=='float': # Use kNN regressor
            decoder = knn_reg(metric='euclidean', n_neighbors=params['num_k'][optimal_k]).fit(train_embedding, shuffled_var)
            shuffled_test_prediction = decoder.predict(test_embedding)
            
            if isCircular:
                shuffled_error[shuffle_i] = np.median(abs((var2predict[data['testingFrames']]-shuffled_test_prediction+180)%360-180))
            else:
                shuffled_error[shuffle_i] = MAE(var2predict[data['testingFrames']], shuffled_test_prediction)
        else: # Use kNN classifier
            decoder = knn_class(metric='euclidean', n_neighbors=params['num_k'][optimal_k]).fit(train_embedding, shuffled_var)
            #prediction = decoder.predict(test_embedding)
            #error_stats[i] = np.nan # could use f1 score? But then opposite of error

        shuffled_score[shuffle_i] = decoder.score(test_embedding,var2predict[data['testingFrames']])
        
    decoding_zscore = (decoding_score-np.mean(shuffled_score))/np.std(shuffled_score)
    decoding_pvalue = np.sum(shuffled_score>decoding_score)/params['num_surrogates']
    shuffled_error = np.nanmean(shuffled_error)
    return decoding_score, decoding_zscore, decoding_pvalue, decoding_error, shuffled_error, test_prediction

def predict_embedding(data, params, embedding):
    np.random.seed(params['seed'])
    spatial_prediction_stats = np.zeros(len(params['num_k']))*np.nan
    retrospective_time_prediction_stats = np.zeros(len(params['num_k']))*np.nan
    prospective_time_prediction_stats = np.zeros(len(params['num_k']))*np.nan
    retrospective_distance_prediction_stats = np.zeros(len(params['num_k']))*np.nan
    prospective_distance_prediction_stats = np.zeros(len(params['num_k']))*np.nan
    heading_prediction_stats = np.zeros(len(params['num_k']))*np.nan
    speed_prediction_stats = np.zeros(len(params['num_k']))*np.nan

    if 'LT_direction' in data:
        data['heading'] = data['LT_direction'] # TODO ensure this still provides a meaningful decoding in 1D environments
    
    # Find optimal k
    for i, num_k  in enumerate(params['num_k']):
        spatial_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(data['position'][data['trainingFrames']], embedding[data['trainingFrames']])
        spatial_prediction_stats[i] = spatial_decoder.score(data['position'][data['testingFrames']], embedding[data['testingFrames']])
        
        retrospective_time_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(data['elapsed_time'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
        retrospective_time_prediction_stats[i] = retrospective_time_decoder.score(data['elapsed_time'][data['testingFrames']].reshape(-1, 1), embedding[data['testingFrames']])

        prospective_time_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(data['time2stop'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
        prospective_time_prediction_stats[i] = prospective_time_decoder.score(data['time2stop'][data['testingFrames']].reshape(-1, 1), embedding[data['testingFrames']])

        retrospective_distance_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(data['distance_travelled'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
        retrospective_distance_prediction_stats[i] = retrospective_distance_decoder.score(data['distance_travelled'][data['testingFrames']].reshape(-1, 1), embedding[data['testingFrames']])

        prospective_distance_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(data['distance2stop'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
        prospective_distance_prediction_stats[i] = prospective_distance_decoder.score(data['distance2stop'][data['testingFrames']].reshape(-1, 1), embedding[data['testingFrames']])

        heading_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(data['heading'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
        heading_prediction_stats[i] = heading_decoder.score(data['heading'][data['testingFrames']].reshape(-1, 1), embedding[data['testingFrames']])

        speed_decoder = knn_reg(metric='euclidean', n_neighbors=num_k).fit(data['velocity'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
        speed_prediction_stats[i] = speed_decoder.score(data['velocity'][data['testingFrames']].reshape(-1, 1), embedding[data['testingFrames']])

    # use the ideal k num_neighbors
    spatial_decoder = knn_reg(metric='euclidean', n_neighbors=params['num_k'][np.argmax(spatial_prediction_stats)]).fit(data['position'][data['trainingFrames']], embedding[data['trainingFrames']])
    retrospective_time_decoder = knn_reg(metric='euclidean', n_neighbors=params['num_k'][np.argmax(retrospective_time_prediction_stats)]).fit(data['elapsed_time'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
    prospective_time_decoder = knn_reg(metric='euclidean', n_neighbors=params['num_k'][np.argmax(prospective_time_prediction_stats)]).fit(data['time2stop'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
    retrospective_distance_decoder = knn_reg(metric='euclidean', n_neighbors=params['num_k'][np.argmax(retrospective_distance_prediction_stats)]).fit(data['distance_travelled'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
    prospective_distance_decoder = knn_reg(metric='euclidean', n_neighbors=params['num_k'][np.argmax(prospective_distance_prediction_stats)]).fit(data['distance2stop'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
    heading_decoder = knn_reg(metric='euclidean', n_neighbors=params['num_k'][np.argmax(heading_prediction_stats)]).fit(data['heading'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
    speed_decoder = knn_reg(metric='euclidean', n_neighbors=params['num_k'][np.argmax(speed_prediction_stats)]).fit(data['velocity'][data['trainingFrames']].reshape(-1, 1), embedding[data['trainingFrames']])
    
    # predict manifold space using each behavioral variable
    spatial_prediction = spatial_decoder.predict(data['position'])
    retrospective_time_prediction = retrospective_time_decoder.predict(data['elapsed_time'].reshape(-1, 1))
    prospective_time_prediction = prospective_time_decoder.predict(data['time2stop'].reshape(-1, 1))
    retrospective_distance_prediction = retrospective_distance_decoder.predict(data['distance_travelled'].reshape(-1, 1))
    prospective_distance_prediction = prospective_distance_decoder.predict(data['distance2stop'].reshape(-1, 1))
    heading_prediction = heading_decoder.predict(data['heading'].reshape(-1, 1))
    speed_prediction = speed_decoder.predict(data['velocity'].reshape(-1, 1))

    return spatial_prediction, retrospective_time_prediction, prospective_time_prediction, retrospective_distance_prediction, prospective_distance_prediction, heading_prediction, speed_prediction

def extract_continuous_HAS(embedding_ref,
                            var_ref,
                            embedding_pred,
                            var_pred):
    
    #TODO use pipeline?
    # Learn mapping between embedding_ref and var_ref
    decoder_var_ref = knn_reg(metric='euclidean', n_neighbors=num_k).fit(embedding_ref, var_ref)

    # Learn mapping between embedding_pred and embedding_ref, 
    decoder_var_pred = decoder_var_ref.predict(lin_reg().fit(embedding_pred, var_pred))

    # Compute score
    HAS = decoder_var_pred.score()

    return HAS