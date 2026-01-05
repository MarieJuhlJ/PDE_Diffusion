from visualize import *

# Define plotting helpers and load data. Keep functions concise and parameterized.
def plot_forecast_loss_vs_steps(df, figsize=(8,5), dir=None, loss_name=None):
    """Reads CSV saved by evaluate.py and plots loss vs forecast step for each Forecast# row."""
    cols = list(df.columns)
    try:
        x = sorted([int(c) for c in cols])
        cols_sorted = [str(c) for c in x]
    except Exception:
        cols_sorted = cols
    plt.figure(figsize=figsize)
    mean = df[cols_sorted].astype(float).mean(axis=0)
    std = df[cols_sorted].astype(float).std(axis=0)
    confidence = 1.96 * std / np.sqrt(len(df))
    plt.plot(range(1, len(cols_sorted)+1), mean.values, marker='o', label='Mean')
    plt.fill_between(range(1, len(cols_sorted)+1), (mean - confidence).values, (mean + confidence).values, alpha=0.3, label='Mean 95% CI')
    plt.xlabel('Forecast step')
    plt.ylabel(f'{loss_name} Loss')
    plt.title('Forecast loss vs forecast steps')
    plt.xticks(range(1, len(cols_sorted)+1))
    plt.legend()
    plt.tight_layout()
    if dir is not None:
        plt.savefig(os.path.join(dir, f'forecast_loss_vs_steps_{loss_name}.png'))
    else:
        plt.savefig('forecast_loss_vs_steps.png')
