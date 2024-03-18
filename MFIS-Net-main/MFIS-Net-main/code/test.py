import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import LADataset
from metric_cal import metric_calculate
from Net import MFISnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(ckp, testfile_path):
    model = MFISnet(1).to(device)
    model.load_state_dict(torch.load(ckp, map_location='cpu'))
    test_dataset = LADataset(testfile_path)
    patient_num = test_dataset.test_patient_num
    name = test_dataset.start_patient
    sum = 0
    patient = 0
    j = 0
    alldice = 0
    valAccu = 0
    allioU = 0
    allsensitivity = 0
    allspecificity = 0
    allhd = 0
    allassd = 0
    d = 0
    a = 0
    b = 0
    c = 0
    e = 0
    h = 0
    s = 0
    count = 0
    test_dataloaders = DataLoader(dataset=test_dataset, batch_size=1)
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloaders), total=len(test_dataloaders), desc='Test'):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            prediction1 = model(inputs)
            prediction2 = torch.round(prediction1)
            img_y = torch.reshape(prediction2, (240, 160)).detach().cpu()
            groundtruth = torch.reshape(labels, (240, 160)).detach().cpu()
            ioU, dice, sensitivity, specificity, hd, assd = metric_calculate(img_y, groundtruth)
            alldice += dice
            allioU += ioU
            allsensitivity += sensitivity
            allspecificity += specificity
            allhd += hd
            allassd += assd
            valAccu += np.mean(img_y.numpy() == groundtruth.numpy())
            if dice < 0.80:
                count += 1
            sum = sum + 1
            while patient_num[j] == 0:
                j += 1
            if sum == patient_num[j]:
                a += valAccu
                d += alldice
                c += allioU
                b += allsensitivity
                e += allspecificity
                h += allhd
                s += allassd
                alldice = alldice / patient_num[j]
                valAccu = valAccu / patient_num[j]
                allioU = allioU / patient_num[j]
                allsensitivity = allsensitivity / patient_num[j]
                allspecificity = allspecificity / patient_num[j]
                allhd = allhd / patient_num[j]
                allassd = allassd / patient_num[j]
                print('{},dice:{},accuracy:{},IoU:{},sensitivity:{},specificity:{},hd:{},assd:{}'.format(
                    name[patient][:-7], alldice, valAccu, allioU, allsensitivity, allspecificity, allhd, allassd))
                j = j + 1
                patient += 1
                sum = 0
                alldice = 0
                valAccu = 0
                allioU = 0
                allsensitivity = 0
                allspecificity = 0
        print('Dice:{},Accuracy:{},IoU:{},Sensitivity:{},Specificity:{},HD:{},ASSD:{}'.format(
            d / (i + 1), a / (i + 1), c / (i + 1), b / (i + 1), e / (i + 1), h / (i + 1), s / (i + 1)))
        print('Number of dice < 0.80: ', count)


if __name__ == '__main__':
    testfile_path = r'E:/PyCharm/Projects/MFIS-net/LAdata/test_2018'
    ckp = r'E:/PyCharm/Projects/MFIS-net/model/best_weights.pth'
    test(ckp, testfile_path)

