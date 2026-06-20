import argparse

import numpy as np
import pandas as pd
import pytest
from ldcov.cli.commands import validate_args
from ldcov.cli.main import main
from tests.helpers import make_symmetric_bm
from ldcov.io.bcor_reader import BcorReader


def test_cli_ld_bm_region(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    df = pd.DataFrame(
        {
            "contig": ["1"] * 6,
            "position": [100 + 10 * i for i in range(6)],
            "ref": ["A"] * 6,
            "alt": ["G"] * 6,
            "idx": list(range(6)),
        }
    )
    variant_index = str(tmp_path / "s.parquet")
    df.to_parquet(variant_index, index=False)
    out = str(tmp_path / "cli_out")

    main(
        [
            "ldcov",
            "--ld-bm",
            "--bm",
            str(tmp_path / "m.bm"),
            "--variant-index",
            variant_index,
            "--region",
            "1:100-150",
            "--out",
            out,
            "--output-format",
            "bcor",
        ]
    )

    back = BcorReader(out + ".bcor").read_corr()
    np.testing.assert_allclose(back, full, atol=1e-4)


def test_cli_npz_format_rejected_without_ld_bm(tmp_path):
    # npz/both are only valid with --ld-bm; --compute-ld must reject them cleanly (not crash).
    # main() catches ValueError inside run_cli and calls sys.exit(1), so expect SystemExit.
    with pytest.raises(SystemExit):
        main(
            [
                "ldcov",
                "--compute-ld",
                "--bgen",
                str(tmp_path / "nonexistent.bgen"),
                "--out",
                str(tmp_path / "x"),
                "--output-format",
                "npz",
            ]
        )


def test_validate_args_npz_rejected_without_ld_bm():
    # Direct unit test of the guard in validate_args.
    args = argparse.Namespace(
        ld_bm=False,
        compute_ld=True,
        precompute_projection=False,
        bgen="dummy.bgen",
        covariates=None,
        projection_matrix=None,
        save_projection=False,
        output_format="npz",
    )
    with pytest.raises(ValueError, match="only valid with --ld-bm"):
        validate_args(args)


def test_cli_storage_options_parsed_and_passed(tmp_path, monkeypatch):
    import ldcov.ld_bm.cli as climod

    captured = {}

    def fake_extract_ld(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(climod, "extract_ld", fake_extract_ld)
    main(
        [
            "ldcov",
            "--ld-bm",
            "--bm",
            "s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm",
            "--variant-index",
            "s3://pan-ukb-us-east-1/x.parquet",
            "--idx-range",
            "0:10",
            "--out",
            str(tmp_path / "o"),
            "--storage-options",
            '{"anon": true}',
        ]
    )
    assert captured["storage_options"] == {"anon": True}


def test_cli_storage_options_invalid_json(tmp_path):
    with pytest.raises(SystemExit):  # main() wraps ValueError -> sys.exit(1)
        main(
            [
                "ldcov",
                "--ld-bm",
                "--bm",
                "s3://bucket/x.bm",
                "--idx-range",
                "0:10",
                "--out",
                str(tmp_path / "o"),
                "--storage-options",
                "{not json}",
            ]
        )


def test_cli_storage_options_non_dict_rejected(tmp_path):
    with pytest.raises(SystemExit):  # main() wraps ValueError -> sys.exit(1)
        main(
            [
                "ldcov",
                "--ld-bm",
                "--bm",
                "s3://bucket/x.bm",
                "--idx-range",
                "0:10",
                "--out",
                str(tmp_path / "o"),
                "--storage-options",
                "[1, 2]",
            ]
        )


def test_validate_args_both_rejected_without_ld_bm():
    # Ensure "both" is also rejected without --ld-bm.
    args = argparse.Namespace(
        ld_bm=False,
        compute_ld=True,
        precompute_projection=False,
        bgen="dummy.bgen",
        covariates=None,
        projection_matrix=None,
        save_projection=False,
        output_format="both",
    )
    with pytest.raises(ValueError, match="only valid with --ld-bm"):
        validate_args(args)
