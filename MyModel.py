import urllib.request
import numpy as np
import torch
import torch.backends.cudnn

from DeepFM import DeepFM

class MyModel(object):

    __flag = False
    __msg = None

    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """
    def __init__(self, fix = 2, url = 'https://shield.mlamp.cn/task/api/file/space/download/b3b84baf1fd59ca01bb3d2e95cde70fe/60288/model.m'):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        self.fix = fix
        self.url = url
        self.loaded = False
        self.model = None

    def load(self):
        print("start download")
        print(self.url)
        urllib.request.urlretrieve(self.url, "model.m")
        print("start loading model")
        self.model = torch.load('model.m', map_location=torch.device('cpu'))
        print("model loaded")

    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """

        if self.deal_parameters(X):
            return "parameter error %s" % self.__msg

        if not self.loaded:
            self.load()

        if self.model:
            self.generate_array()
            x1, x2 = self.generate_torch_data()
            t0 = torch.tensor(x1)
            t1 = torch.tensor(x2)
            result = torch.sigmoid(self.model(t0, t1)).data
            result = self.deal_result(result.numpy())
            return result
        else:
            return "less is more more more more %d" % self.fix

    def get_value(self, parameter):
        value = self.dict.get(parameter)
        if value is None:
            self.__flag = True
            self.__msg = parameter + " is None"
        return value

    def deal_parameters(self, X):
        self.size = X[0]
        self.base_size = X[1]
        self.base_values = X[2]
        self.feature_size = X[3]
        self.feature_values = X[4]

        self.base_values = np.array(self.base_values)
        self.feature_values = np.array(self.feature_values)

        if self.base_values.size != self.base_size:
            self.__flag = True
            self.__msg = 'baseSize check failure'

        if self.feature_values.size != self.feature_size:
            self.__flag = True
            self.__msg = 'featureSize check failure'

        return self.__flag

    def generate_array(self):
        new_arry = np.zeros((self.feature_size, self.base_size))
        for i in range(self.feature_size):
            new_arry[i] = self.base_values

        self.new_array = np.insert(new_arry, 1, self.feature_values, axis=1)
        #print(self.new_array)

    def generate_torch_data(self):
        xi = []
        xv = []

        for size in range(self.feature_size):
            t1, t2 = self.trans(self.new_array[size])
            xi.append(t1)
            xv.append(t2)

        xi = np.array(xi)
        xv = np.array(xv)
        return xi, xv

    def trans(self, aim58):
        t1 = aim58[0:8]
        xi = np.array([ [t1[0]], [t1[1]], [t1[2]], [t1[3]], [t1[4]], [t1[5]], [t1[6]], [t1[7]] ], dtype='long')
        t2_head = [1, 1, 1, 1, 1, 1, 1, 1 ]
        xv = np.array(t2_head + list(aim58[8:]), dtype='float32')
        return (xi, xv)

    def deal_result(self, result):
        dict = {}
        feature = list(self.feature_values)
        result_list = list(result)
        for i in range(self.feature_size):
            dict[feature[i]] = result_list[i]

        result_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        size = len(result_dict)
        if self.size < size:
            return result_dict[:self.size]

        return result_dict


#tt1 =  [[106028, 195092, 10568, 23, 147, 4, 11, 1241, 6.09887064e-01, -3.17003161e-01, -1.83307350e-01, -4.45917211e-02, -4.00365591e-02,  2.60544335e-03, -2.43274420e-02, -1.35902567e-02, -2.06686687e-02, -2.39776302e-04, -8.98106117e-03, -1.32717369e-02, -9.00286250e-03, -9.20017343e-03, -1.12582045e-02, -9.56592243e-03, -5.72999334e-03, -3.99997272e-03, -9.94744524e-03, -6.57328777e-03, -4.06617252e-03, -7.16522615e-03, -3.39697767e-03, -5.05888509e-03, -6.38805423e-03, -6.68853614e-03, -6.55218540e-03, -3.32565443e-03, -7.25812372e-03, -9.18245874e-04,  1.18093006e-03, 3.55028023e-04, -4.88233333e-03, -1.80893322e-03, -3.13342735e-03, -3.14912642e-03, -4.47223382e-03, 8.49320320e-04, -3.30703938e-03, -3.95207189e-06, -3.04178707e-03, -3.35240504e-03, -2.29544588e-03, -2.08881940e-03, -1.75165117e-03, -2.58994359e-03, 5.19961119e-04, -3.13837733e-03, -3.30228242e-03, 3.50067829e-04 ]]
#tt = {"size": 2,"baseSize": 57, "baseValues": [106028, 10568, 23, 147, 4, 11, 1241, 6.09887064e-01, -3.17003161e-01, -1.83307350e-01, -4.45917211e-02, -4.00365591e-02,  2.60544335e-03, -2.43274420e-02, -1.35902567e-02, -2.06686687e-02, -2.39776302e-04, -8.98106117e-03, -1.32717369e-02, -9.00286250e-03, -9.20017343e-03, -1.12582045e-02, -9.56592243e-03, -5.72999334e-03, -3.99997272e-03, -9.94744524e-03, -6.57328777e-03, -4.06617252e-03, -7.16522615e-03, -3.39697767e-03, -5.05888509e-03, -6.38805423e-03, -6.68853614e-03, -6.55218540e-03, -3.32565443e-03, -7.25812372e-03, -9.18245874e-04,  1.18093006e-03, 3.55028023e-04, -4.88233333e-03, -1.80893322e-03, -3.13342735e-03, -3.14912642e-03, -4.47223382e-03, 8.49320320e-04, -3.30703938e-03, -3.95207189e-06, -3.04178707e-03, -3.35240504e-03, -2.29544588e-03, -2.08881940e-03, -1.75165117e-03, -2.58994359e-03, 5.19961119e-04, -3.13837733e-03, -3.30228242e-03, 3.50067829e-04 ], "featureSize": 2, "featureValues": [195092, 195093]}
#tt= [2, 57, [106028, 10568, 23, 147, 4, 11, 1241, 6.09887064e-01, -3.17003161e-01, -1.83307350e-01, -4.45917211e-02, -4.00365591e-02,  2.60544335e-03, -2.43274420e-02, -1.35902567e-02, -2.06686687e-02, -2.39776302e-04, -8.98106117e-03, -1.32717369e-02, -9.00286250e-03, -9.20017343e-03, -1.12582045e-02, -9.56592243e-03, -5.72999334e-03, -3.99997272e-03, -9.94744524e-03, -6.57328777e-03, -4.06617252e-03, -7.16522615e-03, -3.39697767e-03, -5.05888509e-03, -6.38805423e-03, -6.68853614e-03, -6.55218540e-03, -3.32565443e-03, -7.25812372e-03, -9.18245874e-04,  1.18093006e-03, 3.55028023e-04, -4.88233333e-03, -1.80893322e-03, -3.13342735e-03, -3.14912642e-03, -4.47223382e-03, 8.49320320e-04, -3.30703938e-03, -3.95207189e-06, -3.04178707e-03, -3.35240504e-03, -2.29544588e-03, -2.08881940e-03, -1.75165117e-03, -2.58994359e-03, 5.19961119e-04, -3.13837733e-03, -3.30228242e-03, 3.50067829e-04 ], 2, [195092, 195093]]
#aa = MyModel()
#tt = np.array(tt)
#print(aa.predict(tt))

