import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['CONFIG', 'INSTANCE', 'FRAMEWORK', 'TRAINING SPEED'])

df.loc[0] = ('1node-4gpu', '1-p3.8xlarge', 'DDP', 372.581205)
df.loc[1] = ('2node-2gpu', '2-p3.2xlarge', 'DDP', 315.776377)
df.loc[2] = ('4node-4gpu', '4-p3.2xlarge', 'DDP', 190.525508)

df.loc[3] = ('1node-4gpu', '1-p3.8xlarge', 'Horovod', 235.972388)
df.loc[4] = ('2node-2gpu', '2-p3.2xlarge', 'Horovod', 439.631784)
df.loc[5] = ('4node-4gpu', '4-p3.2xlarge', 'Horovod', 274.905384)

df.to_csv("training-speeds.csv")

labels = df["INSTANCE"].unique()
df_ddp = df[(df['FRAMEWORK'] == 'DDP')]
df_hvd = df[(df['FRAMEWORK'] == 'Horovod')]

ddp = df_ddp['TRAINING SPEED'].to_numpy()
hvd = df_hvd['TRAINING SPEED'].to_numpy()

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ddp, width, label='DDP')
rects2 = ax.bar(x + width/2, hvd, width, label='Horovod')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (seconds)')
ax.set_xlabel('AWS EC2 Configuration')
#ax.set_title('Model Training Speed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig(f"training-speeds.png")
plt.show()

