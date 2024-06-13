


Code structure: 
- config
    - settings.py: Key settings & paths, adjustable depending on research questions 
- preprocessing 
    - 001_MESMER_GLMT.py: Computing area-weighted land mean temperature from MESMER grid-point-level temperature output
    - 002_GLMT_characteristics.py: Extracting key characteristics (e.g. end-of-century temperatures or Cumulative exceedance depth)
- main_analysis
    - 001_model_selection.ipynb: Selecting most suitable linear regression model for relating GDP to key GLMT characteristics 



