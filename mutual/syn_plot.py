import json
import numpy as np
import matplotlib.pyplot as plt
# Load the JSON file
with open('/home.stud/quangthi/ws/semester_work/voxelmorp/mse_synthetic.json') as f:
    data_voxelmorp = json.load(f)

with open('/home.stud/quangthi/ws/semester_work/airlab/mse_synthetic.json') as f:
    data_airlab = json.load(f)

with open('/home.stud/quangthi/ws/semester_work/spam/mse_synthetic.json') as f:
    data_spam = json.load(f)

data_lst = {"voxelmorp": data_voxelmorp, "airlab": data_airlab, "spam": data_spam}

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

for key, data in data_lst.items():
    # Extract the values from the dictionary
    trans_loss = data['trans_loss']
    rot_loss = data['rot_loss']

    # Scale x-axis values from 0-100 to -60-60
    x_trans = np.linspace(-5, 5, len(trans_loss))
    x_rot = np.linspace(-60, 60, len(trans_loss))

    # Plot trans_loss
    ax1.plot(x_trans, trans_loss, label=key)
    ax1.set_title('Translation Loss')
    ax1.set_xlabel('Translation in x and y axis (pixel)')
    ax1.set_ylabel('Mean Squared Error')

    # Plot rot_loss
    ax2.plot(x_rot, rot_loss)
    ax2.set_title('Rotation Loss')
    ax2.set_xlabel('Rotation angle (degree)')
    ax2.set_ylabel('Mean Squared Error')

# add legend
ax1.legend()

# add title
fig.suptitle('Registration Loss')
# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
saved_path = "mse_synthetic.png"
plt.savefig(saved_path)

