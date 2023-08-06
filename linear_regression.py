import numpy as np

class LinearRegression:


    def __init__(self):
        pass
    
    def train(self,X,y,learning_rate = 0.01,max_iteration = 10000) :
        self.X = X
        self.y = y
        self.m = len(y)
        self.w = 0
        self.b = 0

        temp_w = self.w
        temp_b = self.b
        
        i = 0

        while i < max_iteration :

            temp_w = self.w - learning_rate * self.gradient_w(self.w , self.b)
            temp_b = self.b - learning_rate * self.gradient_w(self.w , self.b)
            i += 1
            self.w = temp_w
            self.b = temp_b
        

    def cost_function(self , w , b):

        loss = 0
        for i in self.m :
            pred_y  = w * self.X[i] + b
            loss += np.square(pred_y - self.y[i])
        
        cost = loss / (2 * self.m)

        return cost
    
    def gradient_w(self , w , b):

        temp = 0
        for i in range(self.m):
            pred_y  = w * self.X[i] + b
            temp += (pred_y - self.y[i]) * self.X[i]
        
        return temp / self.m
    
    def gradient_b(self , w , b):

        temp = 0
        for i in range(self.m):
            pred_y  = w * self.X[i] + b
            temp += (pred_y - self.y[i])
        
        return temp / self.m

    def predict(self,pred_X) :

        pred = np.array([0 for i in range(self.m)])

        pred = np.dot(self.w,pred_X) + self.b

        return pred
    