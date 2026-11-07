# Third-party notices and acknowledgments

## Project license

MotifHetGNN source code is distributed under the [MIT License](LICENSE), with
copyright attributed to Yuhao Cai. The license file retains the notice from
the [original project repository](https://github.com/wind-whispered/MotifHetGNN/blob/main/LICENSE).

The project license covers the project code. Third-party datasets, executables
and libraries remain subject to their respective terms.

## StatsBomb Open Data

The experiments use [StatsBomb Open Data](https://github.com/hudl/open-data).
Raw data are downloaded separately and are not redistributed in this repository.

Consult the upstream [license](https://github.com/hudl/open-data/blob/master/LICENSE.pdf)
and [attribution instructions](https://github.com/hudl/open-data#terms--conditions)
when using or publishing analysis based on the data. The upstream instructions
ask users who publish or share research to identify StatsBomb as the data source
and use its supplied logo. Software citation through `CITATION.cff` is separate
from this data attribution.

## Optional motif-enumeration software

The external **gtrieScanner** executable is not bundled. Source, documentation
and usage terms are provided by its authors on the
[gtrieScanner project page](https://www.dcc.fc.up.pt/~pribeiro/asd/gtriescanner/).

The repository also includes a pure-Python enumeration backend. See the
[backend configuration instructions](REPRODUCIBILITY.md#motif-enumeration-backend)
for selecting the external executable.

## Python dependencies

The project relies on libraries including PyTorch, PyTorch Geometric, NumPy,
SciPy, pandas, NetworkX, scikit-learn, statsmodels, Matplotlib and seaborn.
Dependency declarations are in [setup.py](setup.py) and
[requirements.txt](requirements.txt).

These libraries are installed separately, not vendored in this repository.
Refer to each library's distribution for its license and notices.

## Figures and assisted development

The architecture schematic is generated from vector primitives by
[scripts/draw_sparc_architecture.py](scripts/draw_sparc_architecture.py).
Empirical plots are generated from recorded numerical outputs. The schematic
does not incorporate downloaded commercial scientific icons or copied journal
illustrations.

AI-assisted tools, including OpenAI Codex, have been used for code drafting,
documentation revision and figure-layout scripting. Scientific interpretation,
validation and release review remain the responsibility of the authors.
