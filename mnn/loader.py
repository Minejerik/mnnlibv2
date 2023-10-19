from mnn.dataset import dataset
import csv

def load_csv(file_name):
  data = dataset()
  with open(file_name,'r') as csvfile:
    reader = csv.reader(csvfile)
    t = 0
    for row in reader:
      if t != 0:
        try:
          gender = row[4]
          gender = 1 if gender == "male" else 2
          survived = int(row[1])
          inp = [gender,int(row[5].split('.')[0])]
          out = [survived]
          data.add_data(inp,out)
        except:
          pass
      t += 1
  return data