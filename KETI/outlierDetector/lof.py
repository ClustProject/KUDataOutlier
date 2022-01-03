"""
    - 예시 코드이며, 각 함수 이름 용도, 파라미터 등은 기술에 따라 달리 정의 부탁드립니다.
"""
class LOF():
    """ LOF outlier Detection Model Manipulation    
    """
    def __init__(self):
        pass
    
    
    def get_model(self, train, method_parameter={'n_neighbors':5}):
        
        """ Train LOF outlier detection model and return trained model

        :param train:  training data
        :type method: DataFrame

        :param method_parameter: ?
        :type method_parameter: json

        :return: outlierDetectionModel
        :rtype: outlierDetectionModel
        
        example
            >>> OD = OutlierDetection()
            >>> OD.setMethod('kde', any_json_parameter)
            >>> outlierDetectionModel = OD.trainModel(train)
        """

        from sklearn.neighbors import LocalOutlierFactor
        # LOF 모델 적합
        n_neighbors = method_parameter['n_neighbors']
        
        model = LocalOutlierFactor(n_neighbors= n_neighbors, novelty=True)
        model.fit(train.values)
        return model  