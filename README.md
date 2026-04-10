I worked on improving the coding side of our protein–nanoparticle interaction prediction
project. The original setup was mainly a single notebook-based prototype. It could
demonstrate the general workflow, but it had several technical weaknesses that made it
fragile and difficult to trust for repeated use. My contribution was to convert that prototype
into a more structured, testable, and reproducible pipeline.
The first major part of my work was restructuring the code. Instead of keeping everything
inside one notebook, I separated the workflow into different modules for data loading,
sequence cleaning, feature extraction, dataset building, benchmarking, and utilities. I also
created a main entry point so the whole workflow can be run from one command. In addition
to that, I made a notebook version of the improved pipeline so the project can still be
demonstrated in notebook form when needed.
A major issue in the original work was brittle dataset handling. The earlier version assumed
the dataset schema without properly checking it, and it loaded only one PPI CSV file instead
of combining the available protein-sequence files. I fixed this by implementing a proper
data-loading layer with schema validation. The updated loader checks whether required
columns exist, reports null counts and duplicates, detects the correct sequence columns,
and combines both positive and negative protein-sequence datasets into one consistent
dataset. This makes the workflow much more reliable and prevents silent failures caused by
wrong assumptions.
Another key part of my contribution was improving protein sequence preprocessing. The
earlier version had problems when unusual residues such as U appeared in the protein
sequences, and malformed sequences were not handled in a structured way. I implemented
a full preprocessing and cleaning system for the protein data. This includes normalization,
uppercase conversion, whitespace removal, invalid-character handling, and logging of
repaired or problematic sequences. In the final run, the pipeline processed more than
146,000 sequence entries and retained more than 10,000 unique cleaned protein
sequences. All preprocessing actions were recorded in exported logs and reports.
I also expanded and stabilized feature extraction. The original notebook used only a few
protein descriptors such as pI, GRAVY, molecular weight, and basic charge category. I kept
those but added more informative descriptors such as sequence length, aromaticity,
instability index, flexibility mean, secondary-structure fractions, and amino-acid composition
values. I also made feature extraction robust by catching and logging failures instead of
letting the entire pipeline crash because of one bad sequence.
Another important part of my work was rebuilding dataset assembly in a clear and traceable
way. I implemented an explicit Cartesian merge between the nanoparticle dataset and the
cleaned protein feature dataset. I also centralized the addition of environmental constants
such as pH, ionic strength, and temperature. The target variable generation was isolated into
its own step, instead of being buried inside notebook cells. This made the workflow easier to
inspect and debug. The final assembled dataset contained over 9.1 million rows, and the
class balance was reported clearly.
I also upgraded the machine learning evaluation section. Previously, the project used only
one model, LinearSVC, with limited evaluation. I implemented benchmarking for four
models: Logistic Regression, LinearSVC, Random Forest, and HistGradientBoosting. I
added held-out testing, cross-validation, confusion matrices, ROC curves, PR curves, and
exported metrics tables. This gave us a much more complete view of how the models
behave under the current pipeline.
On the software quality side, I added reproducibility and testing support. I fixed random
seeds, captured environment versions, created artifact export paths, and added a test suite.
Ten automated tests were run successfully, covering schema validation, ambiguous
sequence-column detection, cleaning policies, duplicate handling, feature extraction failure
handling, dataset assembly, benchmarking, and end-to-end smoke testing. This means the
project is now much easier to rerun and verify compared to the original notebook-only
version.
I also fixed several practical issues encountered during implementation. These included
package/runtime setup issues, merge bugs during feature enrichment, overly large full-data
baseline runs that had to be safely skipped, and rerun failures caused by network-dependent
Kaggle loading. I patched the loader to prefer the local cached dataset path so the project
can rerun in restricted environments more reliably.
One important point I want to make clearly is that although the codebase is now much
stronger technically, the current interaction labels are still synthetic. They are generated
using a heuristic rule based on nanoparticle surface charge and protein pI. Because those
same signals are present in the feature space, the current near-perfect benchmark scores
do not mean the project has achieved true biological prediction performance. So my
contribution mainly strengthens the engineering and ML pipeline side of the work, not the
scientific validity of the label itself.
In summary, my work focused on converting the original fragile notebook into a modular,
validated, reproducible, and benchmarked ML workflow. I handled data loading, schema
validation, protein-sequence preprocessing, feature extraction, dataset assembly,
benchmarking, testing, artifact export, and reproducibility support. This contribution makes
the project much more organized, easier to extend, and more credible from a technical
implementation standpo
