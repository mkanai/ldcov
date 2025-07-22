#!/usr/bin/env python3
"""
Compare benchmark results from multiple ldcov commits.
Generates comparison tables, plots, and identifies performance regressions.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import sys

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.", file=sys.stderr)


class BenchmarkComparator:
    def __init__(self, result_files, output_dir="comparison_results", baseline_commit=None):
        self.result_files = result_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.comparison_data = []
        self.baseline_commit = baseline_commit
        
    def load_results(self):
        """Load all benchmark result files."""
        for file_path in self.result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    commit = data['commit'][:8]  # Use short hash
                    self.results[commit] = data
                    print(f"Loaded results for commit {commit}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}", file=sys.stderr)
    
    def extract_metrics(self):
        """Extract key metrics for comparison."""
        for commit, data in self.results.items():
            commit_info = data.get('commit_info', commit)
            
            for benchmark in data['benchmarks']:
                config = benchmark['config']
                
                # Extract metrics for standard workflow
                if 'standard_workflow' in benchmark['workflows']:
                    workflow_data = benchmark['workflows']['standard_workflow']
                    
                    if 'aggregate' in workflow_data and workflow_data['aggregate'].get('success_rate', 0) > 0:
                        metrics = workflow_data['aggregate']
                        
                        row = {
                            'commit': commit,
                            'commit_info': commit_info,
                            'timestamp': data['timestamp'],
                            'config': config,
                            'samples': benchmark['samples'],
                            'variants': benchmark['variants'],
                            'compression': benchmark['compression'],
                            'file_size_mb': benchmark['file_size_mb'],
                            'median_time': metrics.get('median_time', np.nan),
                            'mean_time': metrics.get('mean_time', np.nan),
                            'std_time': metrics.get('std_time', np.nan),
                            'min_time': metrics.get('min_time', np.nan),
                            'max_time': metrics.get('max_time', np.nan),
                            'median_memory_mb': metrics.get('median_memory_mb', np.nan),
                        }
                        
                        # Add derived metrics if available
                        if 'derived_metrics' in benchmark:
                            row.update({
                                'variants_per_second': benchmark['derived_metrics'].get('variants_per_second', np.nan),
                                'mb_per_second': benchmark['derived_metrics'].get('mb_per_second', np.nan),
                            })
                        
                        self.comparison_data.append(row)
    
    def create_comparison_df(self):
        """Create pandas DataFrame for analysis."""
        self.df = pd.DataFrame(self.comparison_data)
        
        # Sort by timestamp to get chronological order
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['config', 'timestamp'])
        
        # Determine baseline commit
        if self.baseline_commit:
            # User specified baseline
            baseline_to_use = self.baseline_commit
        elif 'master' in self.df['commit'].values:
            # Default to master if available
            baseline_to_use = 'master'
        else:
            # Fallback to earliest commit
            baseline_to_use = None
        
        # Calculate percentage changes
        for config in self.df['config'].unique():
            config_mask = self.df['config'] == config
            config_data = self.df[config_mask].copy()
            
            if len(config_data) > 0:
                # Find baseline row
                if baseline_to_use and baseline_to_use in config_data['commit'].values:
                    baseline = config_data[config_data['commit'] == baseline_to_use].iloc[0]
                else:
                    # Fallback to first commit chronologically
                    baseline = config_data.iloc[0]
                
                for metric in ['median_time', 'median_memory_mb', 'variants_per_second', 'mb_per_second']:
                    if metric in self.df.columns:
                        baseline_value = baseline[metric]
                        if not np.isnan(baseline_value) and baseline_value != 0:
                            if metric in ['variants_per_second', 'mb_per_second']:
                                # Higher is better
                                self.df.loc[config_mask, f'{metric}_pct_change'] = \
                                    ((self.df.loc[config_mask, metric] - baseline_value) / baseline_value) * 100
                            else:
                                # Lower is better
                                self.df.loc[config_mask, f'{metric}_pct_change'] = \
                                    ((self.df.loc[config_mask, metric] - baseline_value) / baseline_value) * 100
    
    def generate_summary_table(self):
        """Generate summary comparison table."""
        summary_file = self.output_dir / "benchmark_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# Benchmark Comparison Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write("## Overall Summary\n\n")
            f.write(f"- Total commits compared: {len(self.results)}\n")
            f.write(f"- Total benchmark configurations: {self.df['config'].nunique()}\n")
            f.write(f"- Total data points: {len(self.df)}\n\n")
            
            # Commit information (sorted by commit date)
            f.write("## Commits\n\n")
            # Extract commit dates from commit_info strings
            commit_dates = {}
            for commit in self.results.keys():
                commit_info = self.results[commit].get('commit_info', '')
                # Try to parse date from commit info string
                # Format: "hash YYYY-MM-DDTHH:MM:SS+TZ:TZ message" (iso-strict)
                # or legacy: "hash YYYY-MM-DD message"
                parts = commit_info.split(' ', 2)
                if len(parts) >= 2:
                    try:
                        # The second part should be the date/timestamp
                        commit_date = pd.to_datetime(parts[1])
                        commit_dates[commit] = commit_date
                    except:
                        # Fall back to benchmark timestamp if parsing fails
                        commit_data = self.df[self.df['commit'] == commit]
                        if not commit_data.empty:
                            commit_dates[commit] = commit_data['timestamp'].min()
                        else:
                            commit_dates[commit] = pd.Timestamp.now()
                else:
                    # Fall back to benchmark timestamp
                    commit_data = self.df[self.df['commit'] == commit]
                    if not commit_data.empty:
                        commit_dates[commit] = commit_data['timestamp'].min()
                    else:
                        commit_dates[commit] = pd.Timestamp.now()
            
            # Sort by commit date
            for commit in sorted(commit_dates.keys(), key=lambda x: commit_dates[x]):
                commit_info = self.results[commit].get('commit_info', commit)
                f.write(f"- **{commit}**: {commit_info}\n")
            f.write("\n")
            
            # Performance comparison by configuration
            f.write("## Performance Comparison\n\n")
            
            # Determine baseline commit for display
            if self.baseline_commit:
                baseline_display = self.baseline_commit
            elif 'master' in self.df['commit'].values:
                baseline_display = 'master'
            else:
                baseline_display = None
            
            if baseline_display:
                f.write(f"**Baseline commit: {baseline_display}**\n\n")
            
            for config in sorted(self.df['config'].unique()):
                config_data = self.df[self.df['config'] == config]
                
                if len(config_data) == 0:
                    continue
                
                f.write(f"### {config}\n\n")
                
                # Create comparison table
                f.write("| Commit | Time (s) | Δ Time | Memory (MB) | Δ Memory | Var/s | Δ Var/s |\n")
                f.write("|--------|----------|--------|-------------|----------|--------|--------|\n")
                
                # Find baseline row
                if baseline_display and baseline_display in config_data['commit'].values:
                    baseline_row = config_data[config_data['commit'] == baseline_display].iloc[0]
                else:
                    # Fallback to first commit chronologically by commit date
                    # Extract commit dates to find the earliest
                    earliest_date = pd.Timestamp.max
                    earliest_commit = None
                    for _, row in config_data.iterrows():
                        commit_info = row.get('commit_info', '')
                        parts = commit_info.split(' ', 2)
                        if len(parts) >= 2:
                            try:
                                # Parse ISO timestamp or legacy date format
                                commit_date = pd.to_datetime(parts[1])
                                if commit_date < earliest_date:
                                    earliest_date = commit_date
                                    earliest_commit = row['commit']
                            except:
                                pass
                    if earliest_commit:
                        baseline_row = config_data[config_data['commit'] == earliest_commit].iloc[0]
                    else:
                        baseline_row = config_data.sort_values('timestamp').iloc[0]
                
                baseline_time = baseline_row['median_time']
                baseline_memory = baseline_row['median_memory_mb']
                baseline_vars = baseline_row.get('variants_per_second', np.nan)
                baseline_commit = baseline_row['commit']
                
                # Sort by commit date to ensure chronological ordering
                # First, extract commit dates for this config's data
                config_commit_dates = {}
                for _, row in config_data.iterrows():
                    commit = row['commit']
                    commit_info = row.get('commit_info', '')
                    # Try to parse date from commit info string
                    # Format: "hash YYYY-MM-DDTHH:MM:SS+TZ:TZ message" or "hash YYYY-MM-DD message"
                    parts = commit_info.split(' ', 2)
                    if len(parts) >= 2:
                        try:
                            # Parse ISO timestamp or legacy date format
                            config_commit_dates[commit] = pd.to_datetime(parts[1])
                        except:
                            config_commit_dates[commit] = row['timestamp']
                    else:
                        config_commit_dates[commit] = row['timestamp']
                
                # Sort config data by commit dates
                sorted_commits = sorted(config_commit_dates.keys(), key=lambda x: config_commit_dates[x])
                
                for commit in sorted_commits:
                    row = config_data[config_data['commit'] == commit].iloc[0]
                    commit = row['commit']
                    time_val = row['median_time']
                    memory_val = row['median_memory_mb']
                    vars_val = row.get('variants_per_second', np.nan)
                    
                    # Calculate changes
                    if commit == baseline_commit:
                        time_change = "baseline"
                        memory_change = "baseline"
                        vars_change = "baseline"
                    else:
                        if not np.isnan(time_val) and baseline_time > 0:
                            pct = ((time_val - baseline_time) / baseline_time) * 100
                            time_change = f"{pct:+.1f}%"
                            if pct > 10:
                                time_change = f"**{time_change}** ⚠️"
                        else:
                            time_change = "N/A"
                        
                        if not np.isnan(memory_val) and baseline_memory > 0:
                            pct = ((memory_val - baseline_memory) / baseline_memory) * 100
                            memory_change = f"{pct:+.1f}%"
                        else:
                            memory_change = "N/A"
                        
                        if not np.isnan(vars_val) and baseline_vars > 0:
                            pct = ((vars_val - baseline_vars) / baseline_vars) * 100
                            vars_change = f"{pct:+.1f}%"
                            if pct < -10:
                                vars_change = f"**{vars_change}** ⚠️"
                        else:
                            vars_change = "N/A"
                    
                    f.write(f"| {commit} | {time_val:.2f} | {time_change} | "
                           f"{memory_val:.1f} | {memory_change} | "
                           f"{vars_val:.1f} | {vars_change} |\n")
                
                f.write("\n")
            
            # Regression analysis
            f.write("## Performance Regressions\n\n")
            
            # Determine baseline commit for regression analysis
            if self.baseline_commit:
                baseline_for_regression = self.baseline_commit
            elif 'master' in self.df['commit'].values:
                baseline_for_regression = 'master'
            else:
                baseline_for_regression = None
            
            regressions = []
            for config in self.df['config'].unique():
                config_data = self.df[self.df['config'] == config]
                
                if len(config_data) > 1:
                    # Find baseline row
                    if baseline_for_regression and baseline_for_regression in config_data['commit'].values:
                        baseline = config_data[config_data['commit'] == baseline_for_regression].iloc[0]
                    else:
                        # Fallback to earliest commit by commit date
                        earliest_date = pd.Timestamp.max
                        earliest_commit = None
                        for _, row in config_data.iterrows():
                            commit_info = row.get('commit_info', '')
                            parts = commit_info.split(' ', 2)
                            if len(parts) >= 2:
                                try:
                                    # Parse ISO timestamp or legacy date format
                                    commit_date = pd.to_datetime(parts[1])
                                    if commit_date < earliest_date:
                                        earliest_date = commit_date
                                        earliest_commit = row['commit']
                                except:
                                    pass
                        if earliest_commit:
                            baseline = config_data[config_data['commit'] == earliest_commit].iloc[0]
                        else:
                            baseline = config_data.sort_values('timestamp').iloc[0]
                    
                    # Check all other commits against baseline
                    for _, row in config_data.iterrows():
                        if row['commit'] != baseline['commit']:
                            time_pct = ((row['median_time'] - baseline['median_time']) / baseline['median_time']) * 100
                            
                            if time_pct > 10:  # More than 10% slower
                                regressions.append({
                                    'config': config,
                                    'baseline_commit': baseline['commit'],
                                    'tested_commit': row['commit'],
                                    'time_increase': time_pct,
                                    'baseline_time': baseline['median_time'],
                                    'tested_time': row['median_time']
                                })
            
            if regressions:
                f.write("⚠️ **Warning: Performance regressions detected!**\n\n")
                for reg in sorted(regressions, key=lambda x: x['time_increase'], reverse=True):
                    f.write(f"- **{reg['config']}**: {reg['time_increase']:.1f}% slower "
                           f"({reg['baseline_time']:.2f}s → {reg['tested_time']:.2f}s) "
                           f"from {reg['baseline_commit']} to {reg['tested_commit']}\n")
            else:
                f.write("✅ No significant performance regressions detected.\n")
            
            f.write("\n")
        
        print(f"Summary saved to: {summary_file}")
        return summary_file
    
    def export_csv(self):
        """Export detailed results to CSV."""
        csv_file = self.output_dir / "benchmark_details.csv"
        self.df.to_csv(csv_file, index=False)
        print(f"Detailed results saved to: {csv_file}")
        return csv_file
    
    def export_json(self):
        """Export comparison data as JSON."""
        json_file = self.output_dir / "benchmark_comparison.json"
        
        comparison_json = {
            'generated': datetime.now().isoformat(),
            'commits': list(self.results.keys()),
            'configurations': list(self.df['config'].unique()),
            'summary': {},
            'detailed_results': self.comparison_data
        }
        
        # Add summary statistics
        for config in self.df['config'].unique():
            config_data = self.df[self.df['config'] == config]
            comparison_json['summary'][config] = {
                'commits': len(config_data),
                'median_time_range': [float(config_data['median_time'].min()), 
                                    float(config_data['median_time'].max())],
                'best_commit': config_data.loc[config_data['median_time'].idxmin(), 'commit'],
                'worst_commit': config_data.loc[config_data['median_time'].idxmax(), 'commit'],
            }
        
        with open(json_file, 'w') as f:
            json.dump(comparison_json, f, indent=2)
        
        print(f"JSON comparison saved to: {json_file}")
        return json_file
    
    def create_plots(self):
        """Create visualization plots."""
        if not PLOT_AVAILABLE:
            print("Skipping plots - matplotlib not available")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Time comparison across commits
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for config in sorted(self.df['config'].unique()):
            config_data = self.df[self.df['config'] == config].sort_values('timestamp')
            if len(config_data) > 1:
                ax.plot(config_data['commit'], config_data['median_time'], 
                       marker='o', label=config, linewidth=2, markersize=8)
        
        ax.set_xlabel('Commit', fontsize=12)
        ax.set_ylabel('Median Time (seconds)', fontsize=12)
        ax.set_title('Performance Across Commits', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_file = self.output_dir / "performance_timeline.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance plot saved to: {plot_file}")
        
        # 2. Throughput comparison
        if 'variants_per_second' in self.df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for config in sorted(self.df['config'].unique()):
                config_data = self.df[self.df['config'] == config].sort_values('timestamp')
                if len(config_data) > 1 and not config_data['variants_per_second'].isna().all():
                    ax.plot(config_data['commit'], config_data['variants_per_second'], 
                           marker='o', label=config, linewidth=2, markersize=8)
            
            ax.set_xlabel('Commit', fontsize=12)
            ax.set_ylabel('Variants per Second', fontsize=12)
            ax.set_title('Throughput Across Commits', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_file = self.output_dir / "throughput_timeline.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Throughput plot saved to: {plot_file}")
        
        # 3. Heatmap of performance changes
        if len(self.results) > 1:
            configs = sorted(self.df['config'].unique())
            commits = sorted(self.results.keys())
            
            # Determine baseline commit for heatmap
            if self.baseline_commit and self.baseline_commit in commits:
                baseline_for_heatmap = self.baseline_commit
            elif 'master' in commits:
                baseline_for_heatmap = 'master'
            else:
                baseline_for_heatmap = None
            
            # Create matrix for heatmap
            perf_matrix = np.zeros((len(configs), len(commits)))
            
            for i, config in enumerate(configs):
                config_data = self.df[self.df['config'] == config]
                
                # Find baseline time for this config
                if baseline_for_heatmap and not config_data[config_data['commit'] == baseline_for_heatmap].empty:
                    baseline_time = config_data[config_data['commit'] == baseline_for_heatmap].iloc[0]['median_time']
                else:
                    # Fallback to earliest commit
                    baseline_time = config_data.sort_values('timestamp').iloc[0]['median_time']
                
                for j, commit in enumerate(commits):
                    commit_data = config_data[config_data['commit'] == commit]
                    if not commit_data.empty:
                        time_val = commit_data.iloc[0]['median_time']
                        if baseline_time > 0:
                            pct_change = ((time_val - baseline_time) / baseline_time) * 100
                            perf_matrix[i, j] = pct_change
                        else:
                            perf_matrix[i, j] = np.nan
                    else:
                        perf_matrix[i, j] = np.nan
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create mask for NaN values
            mask = np.isnan(perf_matrix)
            
            sns.heatmap(perf_matrix, 
                       xticklabels=commits,
                       yticklabels=configs,
                       annot=True, 
                       fmt='.1f',
                       cmap='RdYlGn_r',
                       center=0,
                       mask=mask,
                       cbar_kws={'label': 'Performance Change (%)'},
                       annot_kws={'size': 10})
            
            ax.set_title('Performance Change Heatmap (% vs Baseline)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Commit', fontsize=12)
            ax.set_ylabel('Configuration', fontsize=12)
            plt.tight_layout()
            
            plot_file = self.output_dir / "performance_heatmap.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Heatmap saved to: {plot_file}")
    
    def run_comparison(self):
        """Run the full comparison analysis."""
        print("Loading benchmark results...")
        self.load_results()
        
        if not self.results:
            print("No results loaded. Exiting.")
            return
        
        print("Extracting metrics...")
        self.extract_metrics()
        
        if not self.comparison_data:
            print("No comparison data extracted. Exiting.")
            return
        
        print("Creating comparison dataframe...")
        self.create_comparison_df()
        
        print("Generating outputs...")
        self.generate_summary_table()
        self.export_csv()
        self.export_json()
        self.create_plots()
        
        print(f"\nComparison complete! Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare ldcov benchmark results")
    parser.add_argument("results", nargs="+", help="Benchmark result JSON files")
    parser.add_argument("-o", "--output-dir", default="comparison_results",
                        help="Output directory for comparison results")
    parser.add_argument("--baseline", default=None,
                        help="Baseline commit to compare against (default: master if available, otherwise earliest)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip creating plots")
    
    args = parser.parse_args()
    
    # Validate input files
    result_files = []
    for file_pattern in args.results:
        files = list(Path().glob(file_pattern))
        if not files:
            print(f"Warning: No files found matching '{file_pattern}'", file=sys.stderr)
        result_files.extend(files)
    
    if not result_files:
        print("Error: No valid result files found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(result_files)} result files to compare")
    
    # Run comparison
    comparator = BenchmarkComparator(result_files, args.output_dir, baseline_commit=args.baseline)
    
    if args.no_plots:
        global PLOT_AVAILABLE
        PLOT_AVAILABLE = False
    
    comparator.run_comparison()


if __name__ == "__main__":
    main()