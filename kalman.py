
"""
code reference: 
https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48
https://github.com/mabhisharma/Multi-Object-Tracking-with-Kalman-Filter/blob/master/kalmanFilter.py
"""

import numpy as np





"""K.F
Kalman Filter in matrix notation, class
Simply conclusion:
    mitigate the uncertainty by combining the information we do have with a distribution that we feel more confident in.
"""
class KalmanFilter(object):
    
    """
    default parameters that you could set
    """
    def __init__(self, dt=1, stateVariance=50, measurementVariance=50,
                 method="Velocity"):
        super(KalmanFilter, self).__init__()
        self.method = method
        self.stateVariance = stateVariance
        self.measurementVariance = measurementVariance
        self.dt = dt
        self.initModel()

    """init function to initialise the model"""

    def initModel(self):
        if self.method == "Accerelation":
            self.U = 1
        else:
            self.U = 0
            
        # the deafult values below based on the orginal papers idea
        self.A = np.matrix([[1, self.dt, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, self.dt], [0, 0, 0, 1]])

        self.B = np.matrix([[self.dt ** 2 / 2], [self.dt], [self.dt ** 2 / 2],
                            [self.dt]])

        self.H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])
        
        # Covariance matrix
        self.P = np.matrix(self.stateVariance * np.identity(self.A.shape[0]))
        self.R = np.matrix(self.measurementVariance * np.identity(
            self.H.shape[0]))

        # error terms
        self.Q = np.matrix([[self.dt ** 4 / 4, self.dt ** 3 / 2, 0, 0],
                            [self.dt ** 3 / 2, self.dt ** 2, 0, 0],
                            [0, 0, self.dt ** 4 / 4, self.dt ** 3 / 2],
                            [0, 0, self.dt ** 3 / 2, self.dt ** 2]])
        #error terms
        self.W = np.matrix([[self.dt ** 4 / 4, self.dt ** 3 / 2, 0, 0],
                            [self.dt ** 3 / 2, self.dt ** 2, 0, 0],
                            [0, 0, self.dt ** 4 / 4, self.dt ** 3 / 2],
                            [0, 0, self.dt ** 3 / 2, self.dt ** 2]])
        
        # error covariance matrix, which is P_t, p at previous state
        # at first iteration, current state is previous state
        self.P_t = self.P
        
        # X means the state matrix 
        self.X = np.matrix([[0], [1], [0], [1]])

    

    def predict(self):
        """small p represent the matrix has been update with a new pridiction,
            we could add error term w in pridicted state
            
            Matrix A (state transition) update X and P based upon the time that has elapsed
            
            Matrix B applies the acceleration(u) to provide the vlaues to update the position
            and velocity of AX
        """
        
        # predicted State
        self.X_p_t = self.A * self.X  + self.B * self.U 
        #the pridicted error Co-variance
        self.P_p_t = self.A * self.P_t * self.A.T + self.Q
    
        # return np.asarray(self.X_p_t)[0], np.asarray(self.X_p_t)[2]
        return np.asarray(self.X_p_t)[0]

   

    def kalmanGain(self):
        
        """update Kalman gain
        H matrix helps transform the matrix format of P into the format desierd for K matrix
        
        Y is the matrix contains the measurement data, we could update here
        C is a matrix transform to allowed it be summed with Z
        Z is the error term of the measurement
        """
        # we use persudo inverse since inverse would be more accurate
        self.K = self.P_p_t * self.H.T * np.linalg.pinv(
            self.H * self.P_p_t * self.H.T + self.R)
        
        # we assume there all white noise,  as paper state thus we wont do;
        # i.e no error in the measurement
        #Y_t = C * Y_t_m + Z_m
        
    def update(self,Y):
        """update the Process and state matrix

        Args:
            Y (_type_): current measurement
        """
                
        self.X  = self.X_p_t + self.K * (Y - (self.H * self.X_p_t))

        self.P_t = (np.identity(self.P.shape[0]) -
                        self.K * self.H) * self.P_p_t

    def correct(self, Y):
        
        """correct the state based on historical data,
            Y is the current measurement
        """
        self.kalmanGain()
        self.update(Y)
