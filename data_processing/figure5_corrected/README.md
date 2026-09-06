# Figure 5 - corrected represented-group panels

The two original vector PDFs in the parent `data_processing` directory are
reproduced with their overrepresented/underrepresented assignments reversed, as
requested. Original input files and zero-shot are untouched.

## Reproduction

Place the supplied source PDFs at:
- `data_processing/overrepresented_test_results.pdf`
- `data_processing/underrepresented_test_results.pdf`

These inputs must be provided separately; they are not included in this commit.
Run `python data_processing/figure5_corrected/recover_figure5.py` in an environment
with pypdf, pdfplumber, pypdfium2, numpy and Pillow. The script resolves input/output
paths relative to itself. No model, GPU, training or inference is needed.

## Outputs

- `overrepresented_test_results_corrected.pdf`: the original underrepresented-labeled
  artwork with the title Overrepresented Test Set.
- `underrepresented_test_results_corrected.pdf`: the original overrepresented-labeled
  artwork with the title Underrepresented Test Set.
- Matching PNG previews, an extracted 50-point CSV, vector calibration JSON,
  and verification JSON with source/output hashes.
- Outputs and QA renders are generated locally and ignored by Git.
- No zero-shot or combined plot is produced.

## Fidelity

Original PDF drawing commands and embedded DejaVu Sans fonts are preserved,
including markers, line widths, colors, axes and legends. Only title glyphs,
font resources and title centering change. Replacement titles come directly from
the opposite PDF, retaining exact fonts and kerning. Page dimensions remain
504 x 432 points (7 x 6 inches).

At 144 DPI, every pixel below the title band matches its source exactly.
All extracted line vertices and axis rectangles match exactly. Source SHA-256
hashes remain unchanged. Panel-specific legend/color mappings are preserved.

The script maps PDF vertices to chart values using the labeled y-axis grid.
All 50 values recover to four decimal places with less than 4e-10 residual.
This recovers plotted values, not additional precision or original seed data.

## Scientific interpretation

The group reassignment follows the user's correction. Original F1 labels remain.
The PDFs do not establish pooled versus ligand-macro aggregation, seed averaging,
dataset identity or agreement with manuscript Table 2. No data point is adjusted
to match Table 2. The corrected 650M MoLFormer values are 0.6469 overrepresented
and 0.3946 underrepresented.
