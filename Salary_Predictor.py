import numpy as np
import csv
import matplotlib.pyplot as plt

class Linear_Regression:

    # Initializing the Hyper paramters (learning rate and no. of iterations)
    def __init__(self,learning_rate,no_of_iterations):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    # This function is used to train the data set Model Parameters:(X -> Years of experience, Y-> Salary)
    # Number of training examples(no. of training examples) and number of features (the independent variables)
    def fit(self,x,y):
        # transMyList = np.array([myList]).T
        self.m , self.n = x.shape # X is a vector that represents he number of rows and columns (30x1)
        # Initializing randomly assigned values to the parameters
        self.w = np.zeros(self.n) # The weights (or just the one weight in this case) are represented as a vector
        self.b = 0 # The bias is represented as a tensor of value 0
        self.x = x
        self.y= y

        # Applying the gradient descent algorithm for each training instance
        for i in range(self.no_of_iterations):
            self.update_weight()

    def update_weight(self): # Used to update the parameters (setter)
        y_prediction = self.predict(self.x)

        # Calculating the gradients
        dw = -2 * ((self.x.T).dot(self.y -y_prediction) / self.m)
        print(dw)
        db = -2* np.sum(self.y - y_prediction) / self.m
        print(db)

        # Updating the weights
        self.w = self.w-self.learning_rate*dw
        self.b = self.b-self.learning_rate*db


    def predict(self, X): # The final results after each training instance
        return (X.dot(self.w)) + self.b

#################################################################################################################
while True:
    AmountOfIterations = int(input("Please enter the number of iterations: "))

    # Setting the learning rate (ɑ) and the number of learning
    Model = Linear_Regression(0.001,AmountOfIterations)

    # Getting the Data from the CSV file
    myListX = []
    myListY = []
    with open('salary_data.csv','r') as f:
        myFile = csv.reader(f)
        n= -1
        for row in myFile:
            years, salary = row
            n += 1
            if n > 0:
                myListX.append(float(years))
                myListY.append(int(salary))


    # Converting the lists into the required numpy arrays and fitting them in the model
    ArrayX = np.array([myListX]).T
    ArrayY = np.array([myListY]).T
    Model.fit(ArrayX,ArrayY)


    # Using the model to find the predicted values
    PredictedYValues = []
    for value in myListX:
        PredictedYValues.append(Model.predict(np.array(value)).item(0))

    plt.plot(myListX,PredictedYValues,color = 'red')
    plt.scatter(myListX,myListY)
    plt.xlabel('Years')
    plt.ylabel('Salary')
    plt.title("Salary Against Years of experience")
    plt.show()

    AskForContinuation = input("Would you like to see the model with a different number of epochs?(y/n): ").lower()
    if AskForContinuation == "n":
        break
