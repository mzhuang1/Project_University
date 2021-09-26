"""User Profile

from utils import *
from statistic import *

class User(object):
    def __int__(self, device_id, databuffer):
        self.device_id = 0							#device id from hardware
        self.databuffer = pd.DataFrame(pd.Series([]))	#databuffer
        self.clustering_model = []
        self.reduce_dimension_model = []
        self.normalization_coef =[]
        
    def concat_new_data(self, real_power, reactive_power):

        new_data = pd.concat([pd.Series(real_power, name='p'),pd.Series(reactive_power, name='q')], axis=1)

        try:
            self.databuffer = pd.concat([self.databuffer, new_data])
        except:
            self.databuffer = new_data
            print("There is no data, this is the first time to concatenate data stream")
                


    def event_detect(self):
     

        denoised_real_power = wt_denoising(self.databuffer['p'].values)
        denoised_reactive_power = wt_denoising(self.databuffer['q'].values)
        
        #plt.plot(denoised_real_power)
        
        kde_peak_idx = kde(denoised_real_power)
        #print('the peak size is:', kde_peak_idx.size)
        if kde_peak_idx.size < 2:
        
            self.databuffer = self.databuffer.drop(self.databuffer.index[:base_window*0.5])
            #print('databuffer size after dropping is:', self.databuffer['p'].size)
            return
        #print(denoised_real_power)
        trend_real_power, cyclic_real_power = find_trend(self.databuffer['p'].values, denoised_real_power)
        
        trend_reactive_power, cyclic_reactive_power = find_trend(self.databuffer['q'].values, denoised_reactive_power)
        plt.plot(trend_real_power)
        labels = iter_clean_labels(trend_real_power)
        #print(labels)
    
        if np.any(labels[-4*steady_after_window:] == -1):
            #print('the transient is in the end of the window, let us get some new data and do it again')
            return -1
    
        if np.unique(labels[labels!=-1]).size <= 1:
            #print('there is no event in the second filter')
            self.databuffer = self.databuffer.drop(self.databuffer.index[:base_window*0.5])
            return
        
        acf_peak = acf_peak_index(denoised_real_power)
        
        event_loc = find_event_loc(trend_real_power, labels, acf_peak)
        
        
        
        if event_loc.size == 0:
            #print('I did not find any events')
            self.databuffer = self.databuffer.drop(self.databuffer.index[:base_window*0.5])
            return
        
        ex_event_loc = np.array(extend_event_loc(trend_real_power, event_loc, labels))
        #print(ex_event_loc, type(ex_event_loc))
        if ex_event_loc.size <= 1:
            self.databuffer = self.databuffer.drop(self.databuffer.index[:base_window*0.5])
            return
            
        self.databuffer = self.databuffer.drop(self.databuffer.index[:ex_event_loc[-1]])
        
        return  [extract_power(trend_real_power, trend_reactive_power, ex_event_loc)]
    
    def read_model(self):
        cursor = db.model_data.find({"device_id": self.device_id})
        for i, document in enumerate(cursor):
            self.reduce_dimension_model = pickle.loads(document['dimension_reduction_model_autoencoder'])
            self.clustering_model = pickle.loads(document['prediction_model'])
            self.normalization_coef = pickle.loads(document['normalization_coef'])
    
    def predict_save_feature(self, event, time_stamp):
        try:
            read_model()
            reduced_feature = self.reduce_dimension_model.predict(normalize_online(event[0], event[0], 
                                                                                   self.normalization_coef))
            label = self.clustering_model.predict(reduced_feature)
        except:
            #print('there is no model yet')
            label = -100000
        
        db.feature_data.insert_one(
        {
            "device_id": self.device_id,                
            "feature":pickle.dumps(event),
            "cluster_label_index": label,
            "time_stamp": float(time_stamp)
        }    
        )
        return label

class User_offline_training(object):
    def __int__(self, device_id):
        self.device_id = 0  


    def retrieve_data(self):

        cursor = db.feature.find({"device_id":self.device_id})
        for i, document in enumerate(cursor):
            if i == 0:
                feature = np.array(pickle.loads(document['feature'])[0])
            else:
                feature = np.append(feature, np.array(pickle.loads(document['feature'])[0]), axis=0)
        #return feature
        feature = pd.DataFrame(feature)
        return feature[:fixed_window], feature[-fixed_window:]


    def classifiers_offline(self):

        feature_real_power, feature_reactive_power = self.retrieve_data()
        reduce_dimension_model, feature_reduced, normalization_coef = dimension_reduce(feature_real_power, feature_reactive_power)
        clustering_model, labels = h_clustering(feature_reduced)

        return reduce_dimension_model, clustering_model, normalization_coef, labels

    def save_model(self):
        reduce_dimension_model, clustering_model, normalization_coef, labels = self.classifiers_offline()
        result = db.model_data.update_one(
        {"device_id": self.device_id},
        {
            "$set":{
            "dimension_reduction_model_autoencoder":pickle.dumps(reduce_dimension_model),
            "prediction_model":pickle.dumps(clustering_model),
            "normalization_coef":pickle.dumps(normalization_coef),
            "model_labels":pickle.dumps(labels),
            "model_updated_time":datetime.datetime.utcnow(),
            },
        },
        upsert=True
        )
        
        print("finish update model into mongodb, and the number is ", result.matched_count)


    





