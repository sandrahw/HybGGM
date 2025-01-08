
# lstm example based on https://medium.com/@yangwconion/rainfall-runoff-modeling-using-lstm-51a4fddda4a5 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.utils.data as Data
import seaborn as sns

#TODO sklearn has an option to say how much cpu should be used
#TODO NIce value in script for adapting priority of running script, wiki eejit

#data = pd.read_pickle('/home/hausw001/Data/inputDataML/totInputDataBaseFinal.pkl')
general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'
#load the data
data = pd.read_pickle(r'%s\Results\inputDataML\totInputDataBaseFinal.pkl' %(general_path))
data.set_index('date', inplace=True)

trainset = data.loc['1990-01-01':'2010-12-31']
valset = data.loc['2011-01-01':'2017-12-31']   
testset = data.loc['2018-01-01':'2018-12-31']

trainset = trainset.reset_index(drop=True)
valset = valset.reset_index(drop=True)
testset = testset.reset_index(drop=True)

# transform the input features
transformer_x = MinMaxScaler()
transformer_x.fit(trainset.loc[:,['WLHv','RH', 'EV24', 'QMeuse']].values)
x_train = transformer_x.transform(trainset.loc[:,['WLHv','RH', 'EV24', 'QMeuse']].values)
x_valid = transformer_x.transform(valset.loc[:,['WLHv','RH', 'EV24', 'QMeuse']].values)
x_test = transformer_x.transform(testset.loc[:,['WLHv','RH', 'EV24', 'QMeuse']].values)


lag = 3
x_train = np.concatenate([x_train[(i-lag+1):(i+1),:][np.newaxis,:,:]
                          for i in range(lag-1,len(trainset))
                          if not np.isnan(trainset.loc[i,'QRhine'])])
x_valid = np.concatenate([x_valid[(i-lag+1):(i+1),:][np.newaxis,:,:] 
                          for i in range(lag-1,len(valset)) 
                          if not np.isnan(valset.loc[i,'QRhine'])])
x_test = np.concatenate([x_test[(i-lag+1):(i+1),:][np.newaxis,:,:] 
                         for i in range(lag-1,len(testset)) 
                         if not np.isnan(testset.loc[i,'QRhine'])])
# transform the target variable
transformer_y = PowerTransformer(method='box-cox')
y_train = trainset.loc[(lag-1):,'QRhine'].dropna().values.reshape(-1,1)
y_valid = valset.loc[(lag-1):,'QRhine'].dropna().values.reshape(-1,1)
y_test = testset.loc[(lag-1):,'QRhine'].dropna().values.reshape(-1,1)
transformer_y.fit(y_train)

y_train = transformer_y.transform(y_train)
y_valid = transformer_y.transform(y_valid)
y_test = transformer_y.transform(y_test)

x_train.shape, y_train.shape
x_valid.shape, y_valid.shape
x_test.shape, y_test.shape

device = 'cuda' if torch.cuda.is_available() else 'cpu'  
x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor).to(device)
y_valid = torch.from_numpy(y_valid).type(torch.FloatTensor).to(device)
x_test = torch.from_numpy(x_test).type(torch.FloatTensor).to(device)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor).to(device)
loader = Data.DataLoader(
    dataset=Data.TensorDataset(x_train, y_train),
    batch_size = 50,
    shuffle=True)

class MyLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size,\
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        
    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                       torch.zeros(self.num_layers, batch_size, self.hidden_size))
    
    def forward(self, X):
        output, self.hidden = self.lstm(X, self.hidden)
        output = self.fc(output[:,-1,:])
        return output
    
hidden_size = 4
num_layers = 1
model = MyLSTM(hidden_size, num_layers).to(device)
loss_fn = nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 100
min_loss = float('Inf')
loss_epoch = []
for i in range(epochs):
    print(i)
    for batch_x,batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        model.init_hidden(batch_x.shape[0])
        model.hidden = model.hidden[0].to(device), model.hidden[1].to(device)
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        loss.backward()
        optimizer.step()
    model.init_hidden(x_valid.shape[0])
    model.hidden =  model.hidden[0].to(device), model.hidden[1].to(device)
    y_pred = model(x_valid)
    new_loss = loss_fn(y_pred, y_valid).item()
    loss_epoch.append(new_loss)
    if new_loss<min_loss:
        min_loss = new_loss
        #torch.save(model, '/home/hausw001/Scripts/HybGGM/model.pth')
        torch.save(model, '%s/Scripts/HybGGM/model.pth' %(general_path))


#sns.lineplot(x=np.arange(len(loss_epoch)), y=np.array(loss_epoch))

best_model = torch.load('%s/Scripts/HybGGM/model.pth' %(general_path))
best_model.init_hidden(x_test.shape[0])
best_model.hidden =  best_model.hidden[0].to(device), best_model.hidden[1].to(device)
y_pred = best_model(x_test)
r2_score(transformer_y.inverse_transform(y_test.detach().cpu().numpy()), transformer_y.inverse_transform(y_pred.detach().cpu().numpy()))
# 0.815

sns.lineplot(x=np.arange(len(y_test)), y=transformer_y.inverse_transform(y_test.detach().cpu().numpy()).flatten(),legend='brief',label='QRhine_true')
sns.lineplot(x=np.arange(len(y_pred)), y=transformer_y.inverse_transform(y_pred.detach().cpu().numpy()).flatten(),legend='brief',label='QRhine_pred')
