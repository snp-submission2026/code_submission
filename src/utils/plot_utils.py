import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PLOT_PATH

def plot_confusion_matrix(cm, labels):
    labels = [labels[0][0], labels[1][0], labels[2][0], labels[3][:2], labels[4][:3]]
    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12, "weight": "bold"})
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    plt.ylabel("True Label", fontsize=12, fontweight='bold')
    plt.title("Confusion Matrix (All Classes)", fontsize=12, fontweight='bold', pad=12)
    plt.xticks(fontsize=12, fontweight='bold', rotation=45, ha='right')
    plt.yticks(fontsize=12, fontweight='bold', rotation=0)
    plt.tick_params(axis='both', labelsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)  # resize colorbar ticks
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight('bold')       # make colorbar labels bold
    plt.tight_layout()
    plt.savefig(PLOT_PATH / 'confusion_matrix.pdf', bbox_inches='tight')
