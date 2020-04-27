import pickle
import os

folder = '/root/test/CU_Makhos/models_minimax/drive/'
numItersForTrainExamplesHistory = 10

mainfile = folder+"trainhistory.pth.tar.examples"
old_history = pickle.load(open(mainfile, "rb"))
print(len(old_history))

new_history_all = [[]]
for i in range(1,3):
    print("loading file",i)
    subfile = folder+ "trainhistory" + str(i) + ".pth.tar.examples"
    new_history = pickle.load(open(subfile, "rb"))
    new_history_all[0] += new_history[0]
    

for iter_samples in old_history:
    new_history_all.append(iter_samples)
    

# ---delete if over limit---
if len(old_history) > numItersForTrainExamplesHistory:
    print("len(trainExamplesHistory) =", len(old_history),
          " => remove the oldest trainExamples")
    new_history_all = new_history_all[:numItersForTrainExamplesHistory]
    print('Length after remove:', len(new_history_all))
    
filename = os.path.join(folder, 'trainhistorynew.pth.tar'+".examples")
pickle.dump(new_history_all, open(filename, "wb"))
