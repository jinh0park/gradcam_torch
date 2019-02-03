import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ZeroNet
from data_utils import train_valid_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(data_path='data',batch_size=500, epochs=150, num_classes=10, from_set='test'):
    zeronet = ZeroNet(num_classes=num_classes).to(device)

    train_loader, valid_loader = train_valid_loader(data_path=data_path, batch_size=batch_size, from_set=from_set)

    optimizer = optim.Adam(zeronet.parameters(), lr=1e-3)

    zeronet.train()
    for epoch in range(epochs):
        for batch_index, (X, y) in enumerate(train_loader):
            zeronet.zero_grad()
            X, y = X.to(device), y.to(device)
            scores = zeronet(X)
            loss = F.cross_entropy(scores, y)

            loss.backward()
            optimizer.step()

            if batch_index % 1 == 0:
                train_log = 'Epoch {:2d}/{:2d}\tLoss: {:.6f}\tTrain: [{}/{} ({:.0f}%)]'.format(
                epoch, epochs, loss.cpu().item(), batch_index, len(train_loader),
                100. * batch_index / len(train_loader))
                print(train_log, end='\r')
        print()

        correct = 0.
        last_valid_acc = 0.
        with torch.no_grad():
            cnt = 0
            for batch_index, (X, y) in enumerate(valid_loader):
                X, y = X.to(device), y.to(device)
                scores = zeronet(X)
                predict = scores.argmax(dim=-1)
                correct += predict.eq(y.view_as(predict)).cpu().sum()
                cnt += predict.size(0)
            valid_acc = correct.cpu().item()/cnt*100
            print("validation accuracy: {:.2f}%".format(valid_acc))
            last_valid_acc = valid_acc

    SAVE_PATH = 'saved_model'
    torch.save(zeronet.state_dict(), SAVE_PATH)
    with open('valid.txt', 'w') as f:
        f.write(str(last_valid_acc))



if __name__=="__main__":
    main()
