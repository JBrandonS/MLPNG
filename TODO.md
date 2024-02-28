# runs

Rerun combiner for l_128, 2x, slurm ids 8977955 8977969

# file mods

## config
Add lmax_str to base_name in config, update naming
Rename files to match new lmax

# estimator
Figure out a way to make the estimator faster

# trainer
possibly remove the l=0,1 from the dataloader, these are just 0 and would save space and some compute time?

# new things
same fullsky maps instead of alms, look at the attn method for them