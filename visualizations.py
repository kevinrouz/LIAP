import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

def create_custom_colormap():
    """Create a custom colormap that highlights differences better."""
    colors = [(0.95, 0.95, 0.95), (1, 0.8, 0.2), (0.9, 0.3, 0), (0.6, 0, 0)]
    return LinearSegmentedColormap.from_list('custom_heat', colors, N=100)

def enhanced_attention_visualization(sentence, original_attention, pruned_attention, 
                                    entity_position, entity_type, lime_weights=None, 
                                    save_path=None):
    """Create an enhanced visualization comparing original and pruned attention."""
    tokens = sentence.split()
    
    # Create a figure with custom layout
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 4, 1])
    
    # Custom colormap
    cmap = create_custom_colormap()
    
    # Title for the whole figure
    plt.suptitle(f"Attention Analysis for '{tokens[entity_position]}' ({entity_type})", 
                fontsize=16, y=0.98)
    
    # Token Importance Bar Chart (top row)
    ax_importance = plt.subplot(gs[0, :])
    
    # Calculate token importance from pruned attention
    token_importance = pruned_attention.sum(dim=0).numpy()
    
    # Truncate or pad token_importance to match tokens length
    if len(token_importance) > len(tokens):
        token_importance = token_importance[:len(tokens)]
    elif len(token_importance) < len(tokens):
        # Pad with zeros if needed
        token_importance = np.pad(token_importance, (0, len(tokens) - len(token_importance)))
    
    # Normalize for visualization
    if token_importance.max() > 0:
        token_importance = token_importance / token_importance.max()
    
    # Create bar chart
    bars = ax_importance.bar(range(len(tokens)), token_importance, color='skyblue')
    
    # Highlight the entity
    if entity_position < len(bars):
        bars[entity_position].set_color('red')
    
    # Add LIME weights if available
    if lime_weights is not None:
        # Convert LIME weights to the same scale
        lime_values = np.zeros(len(tokens))
        for word, weight in lime_weights:
            for i, token in enumerate(tokens):
                if word == token:
                    lime_values[i] = weight
        
        if abs(lime_values).max() > 0:
            lime_values = lime_values / abs(lime_values).max()
        
        ax_importance.plot(range(len(tokens)), lime_values, 'go-', label='LIME weights')
        ax_importance.legend()
    
    ax_importance.set_title("Token Importance from LIAP")
    ax_importance.set_xticks(range(len(tokens)))
    ax_importance.set_xticklabels(tokens, rotation=45)
    ax_importance.set_ylabel("Importance")
    
    # 2. Original Attention Heatmap
    ax_original = plt.subplot(gs[1, 0])
    sns.heatmap(original_attention.numpy(), xticklabels=tokens, yticklabels=tokens, 
               ax=ax_original, cmap=cmap, square=True)
    ax_original.set_title("Original Attention")
    
    # Add box around entity position
    if entity_position < len(tokens):
        ax_original.add_patch(plt.Rectangle((entity_position, entity_position), 1, 1, 
                                          fill=False, edgecolor='black', lw=2))
    
    # LIAP-Pruned Attention Heatmap
    ax_pruned = plt.subplot(gs[1, 1])
    sns.heatmap(pruned_attention.numpy(), xticklabels=tokens, yticklabels=tokens, 
               ax=ax_pruned, cmap=cmap, square=True)
    ax_pruned.set_title("LIAP-Pruned Attention")
    
    # Add box around entity position
    if entity_position < len(tokens):
        ax_pruned.add_patch(plt.Rectangle((entity_position, entity_position), 1, 1, 
                                         fill=False, edgecolor='black', lw=2))
    
    # Difference Heatmap (bottom row)
    ax_diff = plt.subplot(gs[2, :])
    diff = pruned_attention.numpy() - original_attention.numpy()
    
    # Use diverging colormap for difference
    sns.heatmap(diff, xticklabels=tokens, yticklabels=tokens, 
               ax=ax_diff, cmap='coolwarm', center=0)
    ax_diff.set_title("Difference (LIAP - Original)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
