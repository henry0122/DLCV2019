import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### Only baseline model
base_loss = open('tmp_log/Baseline/baseline_loss.txt', 'r')
base_loss_lines = base_loss.readlines()

base_val_iou = open('tmp_log/Baseline/baseline_val_iou.txt', 'r')
base_val_iou_lines = base_val_iou.readlines()

loss = []
iou = []
for line in base_loss_lines:
    line = line.strip()
    loss.append(float(line))
    
for line in base_val_iou_lines:
    line = line.strip()
    iou.append(float(line))


plt.figure(figsize=(9, 6))
plt.plot(loss)
plt.title('Training loss versus number of training iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('fig/loss.png')
plt.close()

plt.figure(figsize=(9, 6))
plt.plot(iou)
plt.title('IoU score on validation set versus number of training epoch')
plt.xlabel('Epoch')
plt.ylabel('IoU')
# plt.legend(loc='best', fontsize='medium')
plt.savefig('fig/iou_val.png')
plt.close()



