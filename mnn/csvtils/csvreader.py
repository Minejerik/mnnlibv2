import csv

from mnn.csvtils.csvdata import csvdata

class csvreader:
  def __init__(self,file_name:str):
    self.file_name = file_name


  def read(self):
    data = csvdata()