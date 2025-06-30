#!/usr/bin/env python3
"""
YOLOBench Results Visualization Script

This script reads benchmark results from README.md and generates comprehensive graphs
showing YOLOv8 and YOLOv11 performance across all architectures and devices.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def parse_readme_tables(readme_path="README.md"):
    """Parse YOLOv8 and YOLOv11 benchmark tables from README.md"""
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find YOLOv8 table
    yolov8_pattern = r'## yolov8\s*\n\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*\n\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*\n((?:\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*\n)*)'
    yolov8_match = re.search(yolov8_pattern, content)
    
    # Find YOLOv11 table
    yolov11_pattern = r'## yolov11\s*\n\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*\n\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*\n((?:\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*\n)*)'
    yolov11_match = re.search(yolov11_pattern, content)
    
    def parse_table_data(table_text):
        """Parse table text into structured data"""
        if not table_text:
            return pd.DataFrame()
            
        rows = []
        for line in table_text.strip().split('\n'):
            if line.strip() and '|' in line:
                # Clean up the line and split by |
                cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last elements
                if len(cells) >= 7:  # ARCH, CPU/GPU, n, s, m, l, x
                    arch = cells[0].strip()
                    device = cells[1].strip()
                    try:
                        values = [float(cells[i]) for i in range(2, 7)]
                        rows.append({
                            'ARCH': arch,
                            'Device': device,
                            'n': values[0],
                            's': values[1],
                            'm': values[2],
                            'l': values[3],
                            'x': values[4]
                        })
                    except ValueError:
                        # Skip rows with non-numeric values (like N/A)
                        continue
        
        return pd.DataFrame(rows)
    
    yolov8_data = parse_table_data(yolov8_match.group(1) if yolov8_match else "")
    yolov11_data = parse_table_data(yolov11_match.group(1) if yolov11_match else "")
    
    # Add model version column
    if not yolov8_data.empty:
        yolov8_data['Model'] = 'YOLOv8'
    if not yolov11_data.empty:
        yolov11_data['Model'] = 'YOLOv11'
    
    return yolov8_data, yolov11_data

def create_comprehensive_yolo_chart(df, model_name):
    """Create a comprehensive chart for YOLOv8 or YOLOv11 with all CPU/GPU results"""
    if df.empty:
        print(f"No data available for {model_name}")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Prepare data for plotting
    model_sizes = ['n', 's', 'm', 'l', 'x']
    
    # Separate CPU and GPU data
    cpu_data = df[df['Device'] == 'CPU'].copy()
    gpu_data = df[df['Device'] == 'GPU'].copy()
    
    # Define colors for different architectures
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(cpu_data) + len(gpu_data), 1)))
    
    # Plot CPU results (top subplot)
    if not cpu_data.empty:
        x = np.arange(len(model_sizes))
        width = 0.8 / len(cpu_data)  # Adjust width based on number of architectures
        
        for i, (_, row) in enumerate(cpu_data.iterrows()):
            arch_name = row['ARCH']
            values = [row[size] for size in model_sizes]
            
            # Create shorter labels for better readability
            short_arch = arch_name.replace('Apple ', '').replace('Intel(R) ', '').replace(' CPU', '').replace(' @ 3.60GHz', '')
            if len(short_arch) > 20:
                short_arch = short_arch[:20] + '...'
            
            ax1.bar(x + i * width - (len(cpu_data) - 1) * width / 2, 
                   values, width, label=short_arch, 
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Model Size', fontsize=12)
        ax1.set_ylabel('Inference Time (ms)', fontsize=12)
        ax1.set_title(f'{model_name} CPU Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_sizes)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot GPU results (bottom subplot)
    if not gpu_data.empty:
        x = np.arange(len(model_sizes))
        width = 0.8 / len(gpu_data)  # Adjust width based on number of architectures
        
        for i, (_, row) in enumerate(gpu_data.iterrows()):
            arch_name = row['ARCH']
            values = [row[size] for size in model_sizes]
            
            # Create shorter labels for better readability
            short_arch = arch_name.replace('Apple ', '').replace('NVIDIA ', '')
            if len(short_arch) > 20:
                short_arch = short_arch[:20] + '...'
            
            ax2.bar(x + i * width - (len(gpu_data) - 1) * width / 2, 
                   values, width, label=short_arch, 
                   color=colors[len(cpu_data) + i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Model Size', fontsize=12)
        ax2.set_ylabel('Inference Time (ms)', fontsize=12)
        ax2.set_title(f'{model_name} GPU Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_sizes)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
    
    # If only one type of data exists, remove the empty subplot
    if cpu_data.empty:
        ax1.remove()
        fig.suptitle(f'{model_name} GPU Performance', fontsize=16, fontweight='bold')
    elif gpu_data.empty:
        ax2.remove()
        fig.suptitle(f'{model_name} CPU Performance', fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'{model_name} Performance Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'{model_name.lower()}_comprehensive_performance.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def create_combined_line_chart(yolov8_data, yolov11_data):
    """Create a line chart showing performance trends across model sizes"""
    
    # Combine all data
    all_data = pd.concat([yolov8_data, yolov11_data], ignore_index=True)
    
    if all_data.empty:
        print("No data available for line chart")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('YOLO Performance Trends Across Model Sizes', fontsize=16, fontweight='bold')
    
    model_sizes = ['n', 's', 'm', 'l', 'x']
    x_pos = range(len(model_sizes))
    
    # Plot configurations
    plot_configs = [
        ('YOLOv8', 'CPU', axes[0, 0]),
        ('YOLOv8', 'GPU', axes[0, 1]),
        ('YOLOv11', 'CPU', axes[1, 0]),
        ('YOLOv11', 'GPU', axes[1, 1])
    ]
    
    for model, device, ax in plot_configs:
        data_subset = all_data[(all_data['Model'] == model) & (all_data['Device'] == device)]
        
        if data_subset.empty:
            ax.text(0.5, 0.5, f'No {model} {device} data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{model} {device}')
            continue
        
        # Plot lines for each architecture
        for _, row in data_subset.iterrows():
            arch_name = row['ARCH']
            values = [row[size] for size in model_sizes]
            
            # Create shorter labels
            short_arch = arch_name.replace('Apple ', '').replace('Intel(R) ', '').replace('NVIDIA ', '')
            if len(short_arch) > 25:
                short_arch = short_arch[:25] + '...'
            
            ax.plot(x_pos, values, marker='o', linewidth=2, markersize=6, 
                   label=short_arch, alpha=0.8)
        
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title(f'{model} {device} Performance')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_sizes)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yolo_performance_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: yolo_performance_trends.png")

def main():
    """Main function to generate visualizations"""
    print("YOLOBench Results Visualization")
    print("=" * 40)
    
    # Check if README.md exists
    if not Path("README.md").exists():
        print("Error: README.md not found in current directory")
        return
    
    # Parse README tables
    print("Parsing README.md tables...")
    yolov8_data, yolov11_data = parse_readme_tables()
    
    if yolov8_data.empty and yolov11_data.empty:
        print("No benchmark data found in README.md")
        return
    
    print(f"Found {len(yolov8_data)} YOLOv8 entries and {len(yolov11_data)} YOLOv11 entries")
    
    # Set up matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Create comprehensive charts for each model
    if not yolov8_data.empty:
        create_comprehensive_yolo_chart(yolov8_data, "YOLOv8")
    
    if not yolov11_data.empty:
        create_comprehensive_yolo_chart(yolov11_data, "YOLOv11")
    
    # Create combined trend analysis
    create_combined_line_chart(yolov8_data, yolov11_data)
    
    print("\nVisualization complete!")
    print("Generated files:")
    if not yolov8_data.empty:
        print("- yolov8_comprehensive_performance.png")
    if not yolov11_data.empty:
        print("- yolov11_comprehensive_performance.png")
    print("- yolo_performance_trends.png")

if __name__ == "__main__":
    main()
