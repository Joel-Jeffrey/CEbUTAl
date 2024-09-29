m = "Inceptionv3" # Change as per need


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = InceptionV3(num_class).to(device)


loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), 1e-4)
max_epochs = 100
val_interval = 1
auc_metric = ROCAUCMetric()
model_dir = os.path.join(root_dir,"models")
logdir = os.path.join(root_dir,"logs")
s = str(m)+"_cam"
camdir = os.path.join(root_dir,s)
if os.path.exists(camdir):
    shutil.rmtree(camdir)
    
def dirmake(dir_path):
	if os.path.exists(dir_path):
		return
	else:
		os.makedirs(dir_path)
Dirlist = [model_dir,logdir,camdir]
for i in range(len(Dirlist)):
	dirmake(Dirlist[i])

# print("\n\n",model,"\n")
# print("\n\n",model_dir,"\n",logdir,"\n",camdir)

best_metric = 100
best_metric_epoch = -1
epoch_loss_values = []
val_loss_values = []
metric_values = []
writer = SummaryWriter()
low_loss = 100
low_loss_epoch = -1
model_dir = os.path.join(root_dir,"models")
logdir = os.path.join(root_dir,"logs")
temp = "training_results_"+ str(m) + ".txt"
with open(os.path.join(logdir, temp), 'w') as f:
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        c = 0
        with tqdm(total=len(train_loader)) as pbar:
            for batch_data in train_loader:
                step = pbar.update(1)
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = loss_function(outputs, labels)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
                epoch_loss += loss.item()
                c+=1
                epoch_len = len(train_ds) // train_loader.batch_size
                writer.add_scalar("train_loss", loss.item(), (epoch_len * epoch) + c)
                pbar.set_postfix_str(f"train_loss: {loss.item():.4f}")
        epoch_loss /= c
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}", file=f)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                total_val_loss = 0
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
    
                    #Val Loss
                    outputs = model(val_images)
                    loss = loss_function(outputs, val_labels)
                    total_val_loss += loss.item()
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
    
                #AUC Metric
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
    
                #Val loss
                avgvalloss = total_val_loss/len(val_loader)
                val_loss_values.append(avgvalloss)
    
                #Saving best model
                if avgvalloss <= best_metric:
                    best_metric = avgvalloss
                    best_metric_epoch = epoch + 1
                    mvar = "best_metric_model_"+str(m) + ".pth"
                    torch.save(model.state_dict(), os.path.join(model_dir, mvar))
                    print("saved new best metric model", file=f)
                    print("saved new best metric model")
    
                print(f"current epoch: {epoch + 1}        current AUC: {result:.4f}   current Val_loss: {avgvalloss:.4f} "f" current accuracy: {acc_metric:.4f}    "  f" best Val_loss: {best_metric:.4f}    "   f" at epoch: {best_metric_epoch}", file=f)
                print(f"current epoch: {epoch + 1}        current AUC: {result:.4f}   current Val_loss: {avgvalloss:.4f} "f" current accuracy: {acc_metric:.4f}    "  f" best Val_loss: {best_metric:.4f}    "   f" at epoch: {best_metric_epoch}")
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
writer.close()
