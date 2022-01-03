import pandas as pd
from outlierDetector import lof
"""
    - CLUST 과제는 많은 기관과 많은 오픈소스의 활용 및 연결로 진행해야 하므로, 각 모듈들의 독립이 보장되고 쉽게 활용할 수 있게 interface가 설계되어야 합니다.
    - class 의 각 Fucntion 안에 불필요한 코드 지양 (ex> 모델 생성하는데, 결과 파일을 저장하는 코드가 섞여 있는 경우 지양)
    - 편의상 현재 OutlierDetection 클래스 하위에 2개의 Detection 모듈(kde, iforest) 을 정의 하였음- 현재 제공된 코드 수준은 외부 sklearn 함수를 간단히 호출하는 수준이라 편의상 하위에 정의
    - 각 방법에 대한 적절한 parameter 정의 필요 : 파라미터에 따른 가변적 결과 획득 가능하도록
    - 그러나 현실적으로는 각 방법들에 대해 Class로 설계되어야 함: 
        : 필요에 따라 클래스 생성 및 하위 함수 호출로 원하는 결과를 얻을 수 있도록
        : 현재 lof에 대해서만 작성된 코드를 기반하여 외부 클래스 호출 되도록 수도 코드 넣어 보았음 
    - 각 코드는 Sphinx 스타일 홈페이지 생성을 위한 주석 필수 (향후 Sphinx로 자동 다큐멘테이션 진행 예정임) - 아래 몇개에는 예제 붙였습니다. 최대한 상세하게 기입해주세요.
    - 각 모듈별 스트럭쳐에 대한 플로우 아키텍쳐 생성 필요
    - getOutlierResult에 대한 정의 및 코드 구현 필요

"""
class OutlierDetection():
    """ Outlier Detection Class
    """
    def __init__(self):
        pass
        
    def setMethod(self, method, method_parameter =None):
        """ set outlier detection method and related method parameter

        :param method:  one of methods. method list = ['kde', 'IForest', 'LOF']
        :type method: string
        :param method_parametet:  method parameter that affects outlier detection model building
        :type method: json

        """

        self.method = method
        self.method_parameter = method_parameter
        
    def trainModel(self, train):
        """ Train outlier detection model and return trained model

        :param train:  training data
        :type method: DataFrame

        :return: outlierDetectionModel
        :rtype: outlierDetectionModel
        
        example
            >>> OD = OutlierDetection()
            >>> OD.setMethod('kde', any_json_parameter)
            >>> outlierDetectionModel = OD.trainModel(train)
        """

        if self.method == 'kde':
            outlierDetectionModel = self.kde(train, self.method_parameter)
        elif self.method == 'IForest':
            outlierDetectionModel = self.iforest(train, self.method_parameter)
        elif self.method == 'LOF':
            outlierDetectionModel = lof.LOF().get_model(train, self.method_parameter)
        # 적절한 확장 필요
        else:
            outlierDetectionModel = self.kde(train, self.method_parameter)
        
        return outlierDetectionModel
    
    def getScore(self, model, testdata):
        """ Train outlier detection model

        :param model:  outlierDetectionModel
        :type method: outlierDetectionModel
        :param testdata:  test Data
        :type testdata: DataFrame

        :return: score
        :rtype: DataFrame
        
        example
            >>> score = OutlierDetection().getScore(model, testdata)
        """

        score = - 1.0 * model.score_samples(testdata.values)
        score = pd.DataFrame(score, index=testdata.index, columns=['score'])
        
        return score
        
    def getOutlierResult(self):
        # 최종 결과물에 대한 정의 및 결과 획득 함수 생성 필요

        """ Get Information about whether the input data is outlier or not.

        :param ?:  ?
        :type ?: ?

        :return: result
        :rtype: DataFrame (Boolean??)
        
        example
            >>> result = 
        """

        pass
        
    ########### Model Training
    #KDE, Iforest 필요에 따라 클래스화 진행해야 함
    def kde(self, train, method_parameter=None):
        """ build KDE model

        :param ?:  ?
        :type ?: ?

        :return: model
        :rtype: model
        
        example
            >>> model = OutlierDetection().kde(train, method_parameter)
        """
        from sklearn.neighbors import KernelDensity
        # KDE 기반 분포 추정
        model = KernelDensity(kernel='gaussian', bandwidth=0.2)
        model.fit(train.values)
        return model
    
    def iforest(self, train, method_parameter=None):
        """ build iforest model

        :param ?:  ?
        :type ?: ?

        :return: model
        :rtype: model
        
        example
            >>> model = OutlierDetection().iforest(train, method_parameter)
        """

        from sklearn.ensemble import IsolationForest
        # IForest 모델 적합
        model = IsolationForest(random_state=42)
        model.fit(train.values)
        return model
