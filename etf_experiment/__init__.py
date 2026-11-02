"""ETF household decision experiment.

A controlled-condition study of what makes Chinese households move money out of
deposits into money-market funds, bond funds and equity ETFs. One fixed panel of
10,000 household personas runs every condition; a decision model scores all
candidate products for each household, and the resulting probabilities are
compared across conditions and across models.

Package layout
--------------
etf_experiment.paths      where data, tasks, results and the MAS engine live
etf_experiment.io         jsonl / csv readers and writers
etf_experiment.options    the A-F product menu, conditions and amount arithmetic
etf_experiment.tasks      render household text into persona_choice task rows
etf_experiment.stats      weighted means and household-clustered bootstrap
etf_experiment.launch     start the MAS engine's lanes for one model
etf_experiment.tables     build the baseline / clause / subgroup tables
etf_experiment.figures    the report figures
"""

__version__ = "1.0.0"
