import json
import matplotlib.pyplot as plt
import numpy as np
import ipdb

# Load the JSON data from the file
with open('/home.stud/quangthi/ws/semester_work/voxelmorp/20231128-180552_train.json') as f:
    data = json.load(f)

data = np.array(data)
# ipdb.set_trace()

# Extract the relevant data from the JSON
x = data[:, 1]
y = data[:, 2]

# Plot the data
plt.plot(x, y)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Voxelmorph Training Loss')

# Save the plot as an image
plt.savefig('training_plot.png')

print('Time for each epoch in hour: ', (data[-1, 0] - data[0,0]) / len(data)/3600) # 0.37204445868730546hour
print('Total training time in hour: ', (data[-1, 0] - data[0,0]) /3600)
# Display the plot
plt.show()
