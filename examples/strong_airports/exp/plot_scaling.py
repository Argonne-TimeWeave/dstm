#!/usr/bin/env python3
"""Plot strong scaling results from CSV."""

import csv
import sys
import matplotlib.pyplot as plt

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "scaling_results.csv"

    ranks = []
    elapsed = []
    speedup = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'FAILED' in row['elapsed_seconds']:
                continue
            ranks.append(int(row['ranks']))
            elapsed.append(float(row['elapsed_seconds']))
            speedup.append(float(row['speedup']))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Elapsed Time
    axes[0].plot(ranks, elapsed, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Ranks', fontsize=12)
    axes[0].set_ylabel('Elapsed Time (seconds)', fontsize=12)
    axes[0].set_title('Elapsed Time vs Ranks', fontsize=14)
    axes[0].set_xticks(ranks)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Speedup
    axes[1].plot(ranks, speedup, 'go-', linewidth=2, markersize=8, label='Actual')
    axes[1].plot(ranks, ranks, 'r--', linewidth=1.5, label='Ideal')
    axes[1].set_xlabel('Number of Ranks', fontsize=12)
    axes[1].set_ylabel('Speedup', fontsize=14)
    axes[1].set_title('Strong Scaling Speedup', fontsize=14)
    axes[1].set_xticks(ranks)
    axes[1].set_yticks(ranks)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scaling_results.png', dpi=150)
    print(f"Saved: scaling_results.png")
    plt.show()

if __name__ == '__main__':
    main()
