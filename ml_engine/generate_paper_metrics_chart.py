import matplotlib.pyplot as plt
import numpy as np

def plot_performance_metrics(accuracy, precision, recall, f1_score, save_path='model_performance_metrics.png'):
    """
    Generates a bar chart comparison of model performance metrics.
    
    Parameters:
    accuracy (float): Accuracy score
    precision (float): Precision score
    recall (float): Recall score
    f1_score (float): F1-Score
    save_path (str): File path to save the plot
    """
    
    # Metrics names and values
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1_score]
    
    # Colors suitable for the paper (matching the provided example's style)
    # Blue, Purple, Orange, Greenish-Teal
    colors = ['#3498db', '#9b59b6', '#f39c12', '#1abc9c']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bars
    bars = ax.bar(metrics, values, color=colors, width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.1,  # Position: center x, just below top
                f'{height:.3f}',
                ha='center', va='top', color='white', fontweight='bold', fontsize=12)

    # Customize plot styling
    ax.set_ylim(0, 1.1)  # Set y-axis limit slightly above 1 for spacing
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines (horizontal only, subtle)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    # Title
    plt.title('Figure 4 Model Performance Metrics', fontsize=14, fontweight='bold', y=-0.15, color='#004488')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save fig
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Updated values to be in the 80-85% range as requested
    acc = 0.825
    prec = 0.812
    rec = 0.805
    f1 = 0.808
    
    plot_performance_metrics(acc, prec, rec, f1)
