from mnn.trainer import trainer
from mnn.utils import list_sub

class trainerbackprop(trainer):
    def start_train(self):
        inps = self.data.inps
        outs = self.data.outs
        wanted = []
        for i,o in zip(inps,outs):
            temp = self.net.run(i)
            r = list_sub(o,temp)
            wanted.append(r)
        print(wanted)
