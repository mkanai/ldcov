"""
Entry point for the ldcov command-line interface.

This module provides the main entry point when running ldcov as a module.
"""

# Ultra-minimal entry point - defer all imports
def main():
    """Entry point for ldcov CLI."""
    import sys
    
    # Handle quick options without any imports
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--version', '-V']:
            print("ldcov 0.1.0")
            return
        elif sys.argv[1] in ['--help', '-h'] and len(sys.argv) == 2:
            print("ldcov: Compute linkage disequilibrium with optional covariate adjustment.")
            print("\nUsage: ldcov --bgen FILE [options]")
            print("\nFor full help, use: ldcov --help --verbose")
            print("\nCommon options:")
            print("  --bgen FILE               BGEN genotype file")
            print("  --compute-ld              Compute LD matrix")
            print("  --out PREFIX              Output file prefix")
            print("  -c, --covariates FILE     Covariate file for adjustment")
            print("  --region CHR:START-END    Genomic region to analyze")
            print("  --progress                Show progress bar")
            print("  --help, -h                Show help message")
            print("  --version, -V             Show version")
            return
    
    # Only import the actual CLI when needed
    from .cli.main import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()