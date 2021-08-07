'''
This node computes the average time each epoch takes to
complete in seconds across varying bucket sizes
'''

import pandas as pd
import glob
import json
import matplotlib.pyplot as plt

# 1. Load in files and append the bucket to each record

training_speeds = glob.glob(f"../../ddp/2node-2gpu-bucket/node0/*/train-info.json")

df = pd.DataFrame()

for idx, file in enumerate(training_speeds):
  with open(file) as f:
    # loads JSON as a 1 row dataframe
    jsonData = json.load(f)
    this_df = pd.DataFrame(jsonData, index=[idx])

    # appends bucket name from directory to dataframe as new col
    bucket_name = (file.split("/")[5])
    bucket_num = int(bucket_name.split("-")[1])
    this_df['bucket'] = bucket_num

    # append this 1 row dataframe to master dataframe
    df = df.append(this_df)

# 2. Take the bucket and elapsed training time columns from dataframe
df = df[['elapsed_time', 'bucket']]
df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], downcast="float")
df = df.sort_values(by=['bucket'])
df.to_csv("training-speeds-by-bucket.csv")

ax = df.plot.bar(x='bucket', y='elapsed_time', rot=0,
                 xlabel="Bucket Size (MB)", ylabel = "Time (seconds)",
                 label="Training Speed")

ax.set_ylim([300, 322])
plt.axhline(y = 315.77, color = 'black', linestyle = '-',
            label="Min. Speed: 315.77s")

plt.legend(loc = 'lower right', framealpha=1)
plt.savefig(f"training-speeds-by-bucket.png")
plt.show()