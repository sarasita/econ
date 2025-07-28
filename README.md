


Code structure: 
- config
    - settings.py: Key settings & paths, adjustable depending on research questions 
- preprocessing 
    - 001_GMT_characteristics.py: Extracting key characteristics (e.g. end-of-century temperatures or Cumulative exceedance depth) from raw GMT temperature data 
- main_analysis
    - analysis 
        - 001_processing.ipynb: Extracting relevant GDP data to save computational time
        - 010_clustering.ipynb: Executing clusterin analysis 
        - 020_inequality_and_irreversibility.ipynb: checking climate impacts on between-country inequality and identifying constraints on reversibility of economic impacts
        - 030_regionalisation.py: Quantifying the impacts regionalisation and variability have on GDP 
    - plots
        - 100_until-peak-warming_plots.ipynb: generating Fig. 2 in main manuscript 
        - 101_until-peak-warming_tables.ipynb: generating Latex Table related to Fig. 2 
        - 101_beyond-peak-warming_plots.ipynb: generating Fig. 3 in main manuscript
- stylised plots
    - 000_modelling_graphic.ipynb: generting Fig. 1 in main manuscript
    - 001_appendix_scenarios_graphics.ipynb: generating graphics for the appendix 



