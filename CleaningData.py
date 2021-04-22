import numpy as np

class Data:
  def __init__(self,data):
    self.data=np.genfromtxt(data, delimiter = ',', skip_header=True)
  def CleaningData(self):
    self.x_data=self.data[:,:-1]
    self.y_data=self.data[:,-1]
    self.admidded=self.data[self.y_data == 1]
    self.not_admidded=self.data[self.y_data == 0]
  def ReturnData(self):
    return self.x_data,self.y_data,self.admidded,self.not_admidded
