import numpy as np
import torch
import torch.backends.cudnn
from Network import DeepFM
import connect_s3
import Transform
from importlib import reload

class MyModel(object):

    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """
    def __init__(self, bucket, model_key, network_key, transform_key=None):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        print("bucket: " + bucket)
        print("model key: " + model_key)
        self.loaded = False
        self.model = None
        self.s3 = connect_s3.ConnectS3(bucket, model_key, transform_key, network_key)
        if not transform_key is None:
            print("transform key: " + transform_key)
            reload(Transform)

    def load(self):
        print("start load model")
        self.model = torch.load('./model.m', map_location=torch.device('cpu'))
        print("model loaded")
        self.loaded = True

    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        if not self.loaded:
            self.load()

        if self.model:
            transform = Transform.Transform()
            transform.transform_input(X)
            return transform.transform_output(self.model)
        else:
            return "less is more more more more"

# tt= [2, 57, [106028, 10568, 23, 147, 4, 11, 1241, 6.09887064e-01, -3.17003161e-01, -1.83307350e-01, -4.45917211e-02, -4.00365591e-02,  2.60544335e-03, -2.43274420e-02, -1.35902567e-02, -2.06686687e-02, -2.39776302e-04, -8.98106117e-03, -1.32717369e-02, -9.00286250e-03, -9.20017343e-03, -1.12582045e-02, -9.56592243e-03, -5.72999334e-03, -3.99997272e-03, -9.94744524e-03, -6.57328777e-03, -4.06617252e-03, -7.16522615e-03, -3.39697767e-03, -5.05888509e-03, -6.38805423e-03, -6.68853614e-03, -6.55218540e-03, -3.32565443e-03, -7.25812372e-03, -9.18245874e-04,  1.18093006e-03, 3.55028023e-04, -4.88233333e-03, -1.80893322e-03, -3.13342735e-03, -3.14912642e-03, -4.47223382e-03, 8.49320320e-04, -3.30703938e-03, -3.95207189e-06, -3.04178707e-03, -3.35240504e-03, -2.29544588e-03, -2.08881940e-03, -1.75165117e-03, -2.58994359e-03, 5.19961119e-04, -3.13837733e-03, -3.30228242e-03, 3.50067829e-04 ], 2, [195092, 195093]]
# aa = MyModel("", "", "")
# tt = np.array(tt)
# print(aa.predict(tt))

