import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, device, train_loader, optimizer, loss_f, iteration):
    model.train()
    train_loss, train_acc = AverageMeter(), AverageMeter()
    label_list, predict_list = [],[]

    for i, (inputs, labels) in enumerate(train_loader):
        inputs['csi'], inputs['sp'], labels = inputs['csi'].to(device), inputs['sp'].to(device), labels.to(device)
        predicted = model(inputs['csi'], inputs['sp'])
        predicted = predicted.squeeze()
        labels = labels.to(torch.float32)
        loss = loss_f(predicted, labels)

        predicted = torch.round(predicted)
        label_list.append(labels)
        predict_list.append(predicted)
        train_loss.update(loss.item(), predicted.size(0))
        train_acc.update((predicted == labels).sum().item()/predicted.size(0), predicted.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1

    print(' Train_Loss: ' + str(round(train_loss.avg, 6)), end=" ")
    print(' Train_Accuracy: ' + str(round(train_acc.avg, 6)), end=" ")
    print(' Iteration: ' + str(iteration), end=" ")
    return train_loss.avg, train_acc.avg, iteration


def val_model(model, device, val_loader, loss_f):
    model.eval()
    val_loss, val_acc = AverageMeter(),AverageMeter()
    TP, TN, FP, FN = 0, 0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs['csi'], inputs['sp'], labels = inputs['csi'].to(device), inputs['sp'].to(device), labels.to(device)
            output = model(inputs['csi'], inputs['sp'])
            output = output.squeeze()
            labels = labels.to(torch.float32)
            loss = loss_f(output, labels)
            output = torch.round(output)
            val_loss.update(loss.item(),output.size(0))
            val_acc.update((output == labels).sum().item()/output.size(0), output.size(0))

            # 计算真阳、假阳、真阴、假阴的个数
            TP += ((output == 1) & (labels == 1)).sum().item()
            TN += ((output == 0) & (labels == 0)).sum().item()
            FP += ((output == 1) & (labels == 0)).sum().item()
            FN += ((output == 0) & (labels == 1)).sum().item()
            # progress_bar.set_postfix(loss=val_loss.avg, acc=val_acc.avg)
    print(' Val_loss: ' + str(round(val_loss.avg, 6)))
    print(' Val_Accuracy: ' + str(round(val_acc.avg, 6)), end=" ")
    return val_loss.avg, val_acc.avg, TP, TN, FP, FN


def test_model(model, device, test_loader):
    model.eval()
    results, bina_results = [], []
    with torch.no_grad():
        for i, (input) in enumerate(test_loader):
            torch.cuda.empty_cache()
            input = input.to(device)
            output = model(input)
            output = output.squeeze()
            output = torch.round(output)
            for o in output:
                bina_results.append(o.tolist())

    return results, bina_results