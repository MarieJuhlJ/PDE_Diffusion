import pandas as pd
import matplotlib.pyplot as plt

# Read the metrics CSV file
df = pd.read_csv('logs/exp1-ndcgd/version_0/metrics.csv')

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['train_loss'])
plt.plot(df['step'], df['val_loss'], "x")
plt.legend(['Train Loss', 'Validation Loss'])
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
