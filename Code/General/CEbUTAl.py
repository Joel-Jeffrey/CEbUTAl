# Perform testing using validation set for getting a threshold
mvar = "best_metric_model_"+str(m) + ".pth"
model.load_state_dict(torch.load(os.path.join(model_dir, mvar)))
model.eval()
y_true = []
y_pred = []
prob = []
label = []
iter = -1
label1 = []
label0 = []
gt0 = []
with torch.no_grad():
    for val_data in val_loader:
        val_images, val_labels, fname = (
            val_data[0].to(device),
            val_data[1].to(device),
            val_data[2]
        )
        iter+=1
        pred = model(val_images).argmax(dim=1)
        out = model(val_images)
        probabilities = torch.nn.functional.softmax(out, dim=1)
        for i in range(len(pred)):
          # print(fname[i])
          label.append(int(fname[i].split('/')[5][-1]))
          temp = []
          t = []
          temp.append(fname[i])
          for j in range(2):
              # print(f'  {class_names[j]}: {probabilities[i, j].item()}')
              t.append(probabilities[i,j].item())
          temp.append(t)
          if(label[(i+(len(pred)*iter))]==0):
              gt0.append(temp)
              if pred[i].item() == 0:
                  label0.append(temp)
          elif(label[(i+(len(pred)*iter))]==1):
              if pred[i].item() == 1:
                  label1.append(temp)
          y_true.append(val_labels[i].item())
          y_pred.append(pred[i].item())

def entropy(L):
  templist = []
  for i in range(len(L)):
    e = 0
    for j in range(len(L[i][1])):
        e += float(float(L[i][1][j])*np.log(float(L[i][1][j])))
    e = -float(e)
    templist.append([L[i][0],e])
  return templist

l_entropy = entropy(l)
gt0_entropy = entropy(gt0)
label0_entropy = entropy(label0)
label1_entropy = entropy(label1)


def unc(L):
  templist = []
  for i in range(len(L)):
    uncert = L[i][1]/np.log(0.5)
    u = -1*uncert*100
    templist.append([L[i][0],u])
  return templist

l_unc = unc(l_entropy)
gt0_unc = unc(gt0_entropy)
label0_unc = unc(label0_entropy)
label1_unc = unc(label1_entropy)

def con(L):
  templist = []
  for i in range(len(L)):
    templist.append([L[i][0],max(L[i][1])*100])
  return templist

l_con = con(l)
gt0_con = con(gt0)
label0_con = con(label0)
label1_con = con(label1)

sumunc = 0
c = 0
for i in range(len(label0_unc)):
    sumunc += label0_unc[i][1]
    c+=1

avgunc = sumunc/c

print("Avg unc:",avgunc)

sumcon = 0
c = 0
for i in range(len(label0_con)):
    sumcon += label0_con[i][1]
    c+=1

avgcon = sumcon/c

print("Avg con:",avgcon)
