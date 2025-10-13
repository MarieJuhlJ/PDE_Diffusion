import pandas as pd
import matplotlib.pyplot as plt

# Read the metrics CSV file
df = pd.read_csv('logs/some_name-exp1/version_4/metrics.csv')

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['train_loss'])
plt.xlabel('Training Step')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Time')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
