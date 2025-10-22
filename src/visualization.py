
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import logging
import seaborn as sns

logger = logging.getLogger(__name__)

class StockVisualization:
    
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        plt.style.use('seaborn-v0_8')
        
    def plot_price_vs_predictions(self, data: pd.DataFrame, 
                                 simulation_results: pd.DataFrame,
                                 save_path: Optional[str] = None) -> str:
        
        logger.info("Creating price vs predictions plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(data['timestamp'], data['close_price'], 
                label='Actual Price', color='blue', alpha=0.7)
        
        for _, row in simulation_results.iterrows():
            color = 'green' if row['correct'] else 'red'
            marker = '^' if row['predicted_direction'] == 'up' else 'v'
            ax1.scatter(row['timestamp'], row['current_price'], 
                       color=color, marker=marker, s=100, alpha=0.8)
        
        ax1.set_title('Stock Price with AI Predictions', fontsize=16)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(['Actual Price', 'Correct Prediction', 'Wrong Prediction'])
        ax1.grid(True, alpha=0.3)
        
        simulation_results['cumulative_accuracy'] = simulation_results['correct'].expanding().mean()
        ax2.plot(simulation_results['timestamp'], simulation_results['cumulative_accuracy'], 
                color='purple', linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
        ax2.set_title('Cumulative Prediction Accuracy', fontsize=16)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/price_vs_predictions.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
        
        return save_path
    
    def plot_accuracy_analysis(self, simulation_results: pd.DataFrame,
                              save_path: Optional[str] = None) -> str:
        
        logger.info("Creating accuracy analysis plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        direction_accuracy = simulation_results.groupby('predicted_direction')['correct'].agg(['count', 'mean'])
        direction_accuracy['mean'].plot(kind='bar', ax=ax1, color=['red', 'green'])
        ax1.set_title('Accuracy by Prediction Direction')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Predicted Direction')
        ax1.tick_params(axis='x', rotation=0)
        
        ax2.hist(simulation_results['confidence'], bins=20, alpha=0.7, color='blue')
        ax2.set_title('Prediction Confidence Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        
        simulation_results['confidence_bin'] = pd.cut(simulation_results['confidence'], 
                                                     bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
        conf_accuracy = simulation_results.groupby('confidence_bin')['correct'].mean()
        conf_accuracy.plot(kind='bar', ax=ax3, color='orange')
        ax3.set_title('Accuracy by Confidence Level')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Confidence Level')
        ax3.tick_params(axis='x', rotation=45)
        
        simulation_results['actual_return'] = (simulation_results['next_price'] - simulation_results['current_price']) / simulation_results['current_price']
        correct_returns = simulation_results[simulation_results['correct']]['actual_return']
        wrong_returns = simulation_results[~simulation_results['correct']]['actual_return']
        
        ax4.hist(correct_returns, bins=15, alpha=0.7, label='Correct Predictions', color='green')
        ax4.hist(wrong_returns, bins=15, alpha=0.7, label='Wrong Predictions', color='red')
        ax4.set_title('Return Distribution by Prediction Accuracy')
        ax4.set_xlabel('Daily Return')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/accuracy_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Accuracy analysis saved to {save_path}")
        
        return save_path
    
    def create_performance_dashboard(self, data: pd.DataFrame,
                                   simulation_results: pd.DataFrame,
                                   performance_metrics: Dict,
                                   save_path: Optional[str] = None) -> str:
        
        logger.info("Creating performance dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        ax_main = fig.add_subplot(gs[0, :2])
        ax_main.plot(data['timestamp'], data['close_price'], color='blue', alpha=0.7)
        
        for _, row in simulation_results.iterrows():
            color = 'green' if row['correct'] else 'red'
            marker = '^' if row['predicted_direction'] == 'up' else 'v'
            ax_main.scatter(row['timestamp'], row['current_price'], 
                           color=color, marker=marker, s=60, alpha=0.8)
        
        ax_main.set_title('Stock Price with AI Predictions', fontsize=14)
        ax_main.set_ylabel('Price ($)')
        
        ax_acc = fig.add_subplot(gs[0, 2:])
        simulation_results['cumulative_accuracy'] = simulation_results['correct'].expanding().mean()
        ax_acc.plot(simulation_results['timestamp'], simulation_results['cumulative_accuracy'], 
                   color='purple', linewidth=2)
        ax_acc.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax_acc.set_title('Cumulative Accuracy')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_ylim(0, 1)
        
        ax_metrics = fig.add_subplot(gs[1, 0])
        ax_metrics.axis('off')
        metrics_text = f
        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        ax_dir = fig.add_subplot(gs[1, 1])
        directions = ['Up', 'Down']
        accuracies = [performance_metrics['up_predictions']['accuracy'],
                     performance_metrics['down_predictions']['accuracy']]
        ax_dir.bar(directions, accuracies, color=['green', 'red'], alpha=0.7)
        ax_dir.set_title('Accuracy by Direction')
        ax_dir.set_ylabel('Accuracy')
        ax_dir.set_ylim(0, 1)
        
        ax_conf = fig.add_subplot(gs[1, 2])
        ax_conf.hist(simulation_results['confidence'], bins=15, alpha=0.7, color='blue')
        ax_conf.set_title('Confidence Distribution')
        ax_conf.set_xlabel('Confidence')
        
        ax_timeline = fig.add_subplot(gs[1, 3])
        correct_mask = simulation_results['correct']
        ax_timeline.scatter(range(len(simulation_results)), correct_mask, 
                           c=correct_mask, cmap='RdYlGn', alpha=0.7)
        ax_timeline.set_title('Prediction Timeline')
        ax_timeline.set_xlabel('Prediction #')
        ax_timeline.set_ylabel('Correct (1) / Wrong (0)')
        
        ax_returns = fig.add_subplot(gs[2, :2])
        simulation_results['actual_return'] = (simulation_results['next_price'] - simulation_results['current_price']) / simulation_results['current_price']
        correct_returns = simulation_results[simulation_results['correct']]['actual_return']
        wrong_returns = simulation_results[~simulation_results['correct']]['actual_return']
        
        ax_returns.hist(correct_returns, bins=15, alpha=0.7, label='Correct', color='green')
        ax_returns.hist(wrong_returns, bins=15, alpha=0.7, label='Wrong', color='red')
        ax_returns.set_title('Return Distribution by Prediction Accuracy')
        ax_returns.set_xlabel('Daily Return')
        ax_returns.legend()
        
        ax_scatter = fig.add_subplot(gs[2, 2:])
        colors = ['green' if x else 'red' for x in simulation_results['correct']]
        ax_scatter.scatter(simulation_results['confidence'], simulation_results['actual_return'], 
                          c=colors, alpha=0.6)
        ax_scatter.set_title('Confidence vs Actual Returns')
        ax_scatter.set_xlabel('Prediction Confidence')
        ax_scatter.set_ylabel('Actual Return')
        
        plt.suptitle('AI Stock Prediction Performance Dashboard', fontsize=16, y=0.98)
        
        if save_path is None:
            save_path = f"{self.output_dir}/performance_dashboard.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance dashboard saved to {save_path}")
        
        return save_path