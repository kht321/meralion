"""Generate visualization showing token variant coverage issue."""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Data: base form vs variants
data = {
    'fuck': {
        'base': {'token': 'fuck', 'id': 34024},
        'variants': [
            {'token': 'fucking', 'id': 112487, 'found': True},
            {'token': 'fucker', 'id': 140301, 'found': False},
        ]
    },
    'shit': {
        'base': {'token': 'shit', 'id': 31947},
        'variants': [
            {'token': 'shitting', 'id': 31947, 'found': False},
            {'token': 'bullshit', 'id': 29235, 'found': False},
        ]
    },
    'damn': {
        'base': {'token': 'damn', 'id': 48542},
        'variants': [
            {'token': 'damned', 'id': 9720, 'found': False},
            {'token': 'dammit', 'id': 9720, 'found': False},
        ]
    },
    'piss': {
        'base': {'token': 'piss', 'id': 235263},
        'variants': [
            {'token': 'pissed', 'id': 4002, 'found': True},
            {'token': 'pissing', 'id': 235263, 'found': False},
        ]
    },
    'slut': {
        'base': {'token': 'slut', 'id': 178693},
        'variants': [
            {'token': 'slutty', 'id': 178693, 'found': True},
            {'token': 'sluts', 'id': 8284, 'found': False},
        ]
    },
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Before (only base forms blocked)
ax1.set_title('Initial Guardrail: Base Forms Only (8 keywords)\n0% Blocking Success', 
              fontsize=12, fontweight='bold')
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 4.5)
ax1.set_xlabel('Token Variant Type')
ax1.set_ylabel('Keyword Base Form')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Base\n(Banned)', 'Variants\n(Not Banned)'])
ax1.set_yticks(range(len(data)))
ax1.set_yticklabels(list(data.keys()))
ax1.invert_yaxis()

# Plot base forms (red = banned)
for i, (base_word, info) in enumerate(data.items()):
    ax1.scatter([0], [i], s=800, c='red', alpha=0.7, marker='s', 
                edgecolors='black', linewidths=2, label='Banned' if i == 0 else '')
    ax1.text(0, i, info['base']['token'], ha='center', va='center', 
             fontsize=9, fontweight='bold', color='white')

# Plot variants (green = not banned, so model can generate them)
for i, (base_word, info) in enumerate(data.items()):
    for j, variant in enumerate(info['variants']):
        x_offset = 1 + j * 0.15
        marker = '*' if variant['found'] else 'o'
        color = 'orange' if variant['found'] else 'lightgreen'
        ax1.scatter([x_offset], [i], s=600, c=color, alpha=0.7, marker=marker,
                    edgecolors='black', linewidths=1.5)
        ax1.text(x_offset, i + 0.3, variant['token'], ha='center', va='top', 
                 fontsize=7, style='italic')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Banned token (masked at -∞)'),
    Patch(facecolor='orange', alpha=0.7, edgecolor='black', label='Generated variant (found in dataset)'),
    Patch(facecolor='lightgreen', alpha=0.7, edgecolor='black', label='Other variant (not tested)'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Chart 2: After (base + variants blocked)
ax2.set_title('Fixed Guardrail: Base + Variants (29 keywords)\n30% Blocking Success', 
              fontsize=12, fontweight='bold')
ax2.set_xlim(-0.5, 2.5)
ax2.set_ylim(-0.5, 4.5)
ax2.set_xlabel('Token Variant Type')
ax2.set_ylabel('Keyword Base Form')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Base\n(Banned)', 'Variants\n(Also Banned)'])
ax2.set_yticks(range(len(data)))
ax2.set_yticklabels(list(data.keys()))
ax2.invert_yaxis()

# Plot base forms (red = banned)
for i, (base_word, info) in enumerate(data.items()):
    ax2.scatter([0], [i], s=800, c='red', alpha=0.7, marker='s', 
                edgecolors='black', linewidths=2)
    ax2.text(0, i, info['base']['token'], ha='center', va='center', 
             fontsize=9, fontweight='bold', color='white')

# Plot variants (ALL red = all banned now)
for i, (base_word, info) in enumerate(data.items()):
    for j, variant in enumerate(info['variants']):
        x_offset = 1 + j * 0.15
        marker = '*' if variant['found'] else 's'
        ax2.scatter([x_offset], [i], s=600, c='red', alpha=0.7, marker=marker,
                    edgecolors='black', linewidths=1.5)
        ax2.text(x_offset, i + 0.3, variant['token'], ha='center', va='top', 
                 fontsize=7, style='italic')

# Add legend
legend_elements2 = [
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='All variants banned (masked at -∞)'),
]
ax2.legend(handles=legend_elements2, loc='upper right', fontsize=8)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/guardrails/charts/token_variant_coverage.png', dpi=300, bbox_inches='tight')
print("Created: results/guardrails/charts/token_variant_coverage.png")
