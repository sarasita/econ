
#%%

from pathlib import Path 


#%%

scenarios    = ['ssp119', 'SP', 'Neg', 'LD', 'Ren', 'Ref1p5', 'GS', 'ssp534-over', 'ModAct', 'CurPol']
path_PROJECT = Path("/Users/schoens/Documents/Projekte/Econ/")
path_MESMER  = Path.joinpath(path_PROJECT, "Data/MESMER/")
path_FAIR    = Path.joinpath(path_PROJECT, "Data/FaIR_v2/")
path_GDP     = Path.joinpath(path_PROJECT, "Data/BHM/")


# %%

# OUTPUT 
path_MESMER_fldmean = Path.joinpath(path_PROJECT, "Data/MESMER/fldmean/")