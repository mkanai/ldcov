#!/usr/bin/env python3
"""
Analyze and visualize profiling results from benchmark runs.
"""

import argparse
import json
import os
import pstats
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def analyze_python_profile(prof_file, output_dir):
    """Analyze Python cProfile data and generate reports."""
    print(f"\nAnalyzing Python profile: {prof_file}")
    
    # Load profile stats
    stats = pstats.Stats(str(prof_file))
    stats.strip_dirs()
    stats.sort_stats('cumulative', 'time')
    
    # Generate text report
    report_file = output_dir / f"{prof_file.stem}_analysis.txt"
    with open(report_file, 'w') as f:
        # Summary stats
        f.write("=== PROFILE SUMMARY ===\n")
        f.write(f"Total calls: {stats.total_calls}\n")
        f.write(f"Primitive calls: {stats.prim_calls}\n")
        f.write(f"Total time: {stats.total_tt:.3f} seconds\n\n")
        
        # Top 50 functions by cumulative time
        f.write("=== TOP 50 FUNCTIONS BY CUMULATIVE TIME ===\n")
        stream = stats.stream
        stats.stream = f
        stats.print_stats(50)
        
        # Top 30 callers
        f.write("\n\n=== TOP 30 CALLERS ===\n")
        stats.print_callers(30)
        
        # Top 30 callees
        f.write("\n\n=== TOP 30 CALLEES ===\n")
        stats.print_callees(30)
        
        stats.stream = stream
    
    print(f"  Generated text report: {report_file}")
    
    # Generate call graph with gprof2dot if available
    if subprocess.run(["which", "gprof2dot"], capture_output=True).returncode == 0:
        dot_file = output_dir / f"{prof_file.stem}_callgraph.dot"
        svg_file = output_dir / f"{prof_file.stem}_callgraph.svg"
        
        # Generate dot file
        cmd = ["gprof2dot", "-f", "pstats", str(prof_file), "-o", str(dot_file)]
        subprocess.run(cmd, check=True)
        
        # Convert to SVG
        cmd = ["dot", "-Tsvg", str(dot_file), "-o", str(svg_file)]
        subprocess.run(cmd, check=True)
        
        print(f"  Generated call graph: {svg_file}")
    
    # Generate interactive HTML visualization with snakeviz if available
    if subprocess.run(["which", "snakeviz"], capture_output=True).returncode == 0:
        html_file = output_dir / f"{prof_file.stem}_snakeviz.html"
        
        # Run snakeviz in headless mode to generate static HTML
        cmd = ["snakeviz", "--server", "--port", "0", str(prof_file)]
        # Note: This would normally start a server, but we just want the visualization
        print(f"  To view interactive visualization, run: snakeviz {prof_file}")


def analyze_perf_data(perf_file, output_dir):
    """Analyze perf data and generate reports."""
    print(f"\nAnalyzing perf data: {perf_file}")
    
    # Generate detailed report
    report_file = output_dir / f"{perf_file.stem}_detailed.txt"
    cmd = ["perf", "report", "--stdio", "--no-children", "-i", str(perf_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    with open(report_file, 'w') as f:
        f.write(result.stdout)
    
    print(f"  Generated detailed report: {report_file}")
    
    # Generate annotated source if possible
    annotate_file = output_dir / f"{perf_file.stem}_annotated.txt"
    cmd = ["perf", "annotate", "--stdio", "-i", str(perf_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        with open(annotate_file, 'w') as f:
            f.write(result.stdout)
        print(f"  Generated annotated source: {annotate_file}")


def find_profile_files(profile_dir):
    """Find all profile files in the directory."""
    profile_dir = Path(profile_dir)
    
    files = {
        'python_profiles': list(profile_dir.glob("*_python.prof")),
        'python_stats': list(profile_dir.glob("*_python_stats.txt")),
        'perf_data': list(profile_dir.glob("*_perf.data")),
        'perf_reports': list(profile_dir.glob("*_perf_report.txt")),
        'flame_graphs': list(profile_dir.glob("*_flame.svg"))
    }
    
    return files


def generate_summary_report(profile_dir, output_file):
    """Generate a summary report of all profiling results."""
    files = find_profile_files(profile_dir)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'profile_directory': str(profile_dir),
        'summary': {
            'total_python_profiles': len(files['python_profiles']),
            'total_perf_profiles': len(files['perf_data']),
            'total_flame_graphs': len(files['flame_graphs'])
        },
        'files': {
            'python_profiles': [str(f.name) for f in files['python_profiles']],
            'perf_data': [str(f.name) for f in files['perf_data']],
            'flame_graphs': [str(f.name) for f in files['flame_graphs']]
        }
    }
    
    # Extract key insights from Python stats files
    insights = []
    for stats_file in files['python_stats']:
        with open(stats_file, 'r') as f:
            content = f.read()
            # Extract top function from stats
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'cumulative' in line and i + 2 < len(lines):
                    # Next non-empty line after header should be top function
                    top_func_line = lines[i + 2].strip()
                    if top_func_line:
                        insights.append({
                            'file': stats_file.name,
                            'top_function': top_func_line
                        })
                    break
    
    if insights:
        report['insights'] = insights
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nGenerated summary report: {output_file}")
    
    # Print summary to console
    print("\n=== PROFILING SUMMARY ===")
    print(f"Total Python profiles: {report['summary']['total_python_profiles']}")
    print(f"Total perf profiles: {report['summary']['total_perf_profiles']}")
    print(f"Total flame graphs: {report['summary']['total_flame_graphs']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze profiling results from benchmarks")
    parser.add_argument("profile_dir", 
                        help="Directory containing profile files (e.g., benchmark_results/profiles)")
    parser.add_argument("--output-dir", "-o",
                        help="Output directory for analysis results (default: same as profile_dir)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only generate summary report without detailed analysis")
    args = parser.parse_args()
    
    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        print(f"Error: Profile directory not found: {profile_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else profile_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing profiles in: {profile_dir}")
    print(f"Output directory: {output_dir}")
    
    # Generate summary report
    summary_file = output_dir / "profile_summary.json"
    generate_summary_report(profile_dir, summary_file)
    
    if not args.summary_only:
        # Find and analyze all profile files
        files = find_profile_files(profile_dir)
        
        # Analyze Python profiles
        for prof_file in files['python_profiles']:
            try:
                analyze_python_profile(prof_file, output_dir)
            except Exception as e:
                print(f"  Error analyzing {prof_file}: {e}")
        
        # Analyze perf data
        for perf_file in files['perf_data']:
            try:
                analyze_perf_data(perf_file, output_dir)
            except Exception as e:
                print(f"  Error analyzing {perf_file}: {e}")
        
        print(f"\nAnalysis complete. Results in: {output_dir}")
        
        # Print instructions for viewing results
        print("\n=== VIEWING RESULTS ===")
        print("Text reports: View .txt files in the analysis directory")
        print("Call graphs: Open .svg files in a web browser")
        print("Interactive Python profiles: Run 'snakeviz <profile>.prof'")
        print("Flame graphs: Open *_flame.svg files in a web browser")


if __name__ == "__main__":
    main()