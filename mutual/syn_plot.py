import json
import numpy as np
import matplotlib.pyplot as plt
# Load the JSON file
with open('/home.stud/quangthi/ws/semester_work/airlab/mse_synthetic.json') as f:
    data = json.load(f)

# Extract the values from the dictionary
trans_loss = data['trans_loss']
rot_loss = data['rot_loss']

# Scale x-axis values from 0-100 to -60-60
x_trans = np.linspace(-5, 5, len(trans_loss))
x_rot = np.linspace(-60, 60, len(trans_loss))
# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot trans_loss
ax1.plot(x_trans, trans_loss)
ax1.set_title('Translation Loss')
ax1.set_xlabel('Translation in both axes')
ax1.set_ylabel('Mean Squared Error')

# Plot rot_loss
ax2.plot(x_rot, rot_loss)
ax2.set_title('Rotation Loss')
ax2.set_xlabel('Rotation angle (degree)')
ax2.set_ylabel('Mean Squared Error')


fig.suptitle('Airlab Registration Loss')
# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('mse_synthetic_airlab.png')

# Show the figure
plt.show()
