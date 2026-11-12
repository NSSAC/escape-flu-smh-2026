# General process used Round 20 calibration and projection

## 1. Select parameters and range (initially just using intuition from manual random walk calibration)
## 2. Use analysis tools in "notebooks/sweep" to analyze and find the closest cells to the objection (0-64 and 65+ hosp admissions during 2025-2026 year)
```bash
./run_all_states.sh -i ../../calib_48_states-3/ -o ../../calib_particle_filtering-3 -s "AL  AR  AZ  CA  CO  CT  DC  DE  FL  GA"
```
This will generate sbatches for each state and runs extract -> score -> explore -> top particles, writing to \<out>/\<STATE>/

## 3. Assess how well the particles fit and reconsider the design and bounds.  To generate a new "design" (sampling frame to draw parameter sets from)

```bash
python sweep_multistate_bounds.py -i out/calib48 --add winter_seasonality_scale --prior-sweep out/VA-B-out31 --coverage 0.95
```
This script has lots of arguments that are documented in the comments at the top of it.  Basically it assesses all sates in the input folder, considers the "add" parameters to make a design, using --prior-sweep to inform estimates for the added parameter, and coverage is its tolerance for how many states being "pressed" on a bound necessitate widening it.

The output design is printed to the screen and looks like:
```json
proposed SEARCH_SPACE:
SEARCH_SPACE = {
    "baseline_tau": FloatRange(0.00855282, 0.0192167),   # was 0.0095027-0.019999; extended LOW (3 states pressed)
    "tau_seasonality_scale": FloatRange(0.00414045, 0.0202137),   # was 0.00500949-0.0199892; extended LOW (1 states pressed)
    "hosp_prob_scale": FloatRange(0.65953, 1.83111),   # was 0.700203-1.79937
    "hosp_prob_65p_scale": FloatRange(0.759022, 2.3953),   # was 0.800121-2.39971
    "infection_waning_per_day": FloatRange(0.989985, 0.996975),   # was 0.990149-0.99685
    "winter_seasonality_scale": FloatRange(0.0576951, 2.39269),   # RE-ADDED: implied by the winter misfit of 23 states at the frozen 0.7 (implied 0.09-2.40), unioned with VA-B-out31
}
```
## 4. Generate particles for this design using escape-covid-smh/scripts/explore_params

* Ensure the search space is in sync (copied into) into `scripts/explore_params.py`.
* Next run the parameter sweep with:

```bash
cd scripts
sbatch explore_params.sbatch
```

* When this is done the output files will in scratch.
* Fix the src and final target directory and run rsync to copy.

```bash
cd /scratch/$USER
mkdir -p /project/bii_nssac/biocomplexity/SMH/covid19-round20/CALIB_48_STATES-NNN
rsync -avm --include='*/' --include='output-*.h5' --exclude='*' CALIB_48_STATES-NNN/ /project/bii_nssac/biocomplexity/SMH/covid19-round20/CALIB_48_STATES-NNN
```

## 5. Generate parameter sets for each trajectory for the projections:
```bash
python sweep_replicates.py --from ../../calib_particle_filtering/AL/top_cells.csv --particles 100 --ratio-window 0.95:1.10 -n 100

python sweep_replicates.py --from ../../calib_particle_filtering-3/DE/top_cells.csv --particles 100 --ratio-window 0.95:1.10 -n 100
```

## 6. Project with the generated replicates CSV using escape-covid19-smh/scripts/run_projection.py  

* Ensure the proper regions are selected in `scripts/run_projection.py`.
* Next do the projection runs

```bash
cd scripts
sbatch run_projection.sbatch
```

* Once this is done the output will be in `/scratch`
* Copy it over to `/project`


```bash
cd /scratch/$USER
mkdir -p /project/bii_nssac/biocomplexity/SMH/covid19-round20/PROJ_48_STATES-NNN
rsync -avm --include='*/' --include='output-*.h5' --exclude='*' PROJ_48_STATES-NNN/ /project/bii_nssac/biocomplexity/SMH/covid19-round20/PROJ_48_STATES-NNN
```

## 7. Analyze the full dynamics with the abm_analysis pipeline if you want to double check vaccine effectiveness, fit of deaths, fit of hosp across ages, variability of trajectories and overall hosp averted across scenarios
```bash
./run_analyses.sh -i proj-al-2/ 

Examples:
  ./run_analyses.sh -i test-output33
  ./run_analyses.sh -i /scratch/bl4zc/covid19-round20/proj-al-1 -s AL
  ./run_analyses.sh -i calib_particle_filtering/AL/top_particles -o /tmp/al-analysis

  ./run_analyses.sh -i proj-final-submission/ --max-reps 20 ### limit to 20 reps for "quicker" processing
```
This generates ~30 plots per scenario and a "comparison" across scenarios and also dumps the corresponding CSV files for further analysis / tweaks.  These are placed in an "analysis" foder with "figures" and "data" for each type. 


## 8. Prep for submission.
```bash
 # everything found under test-output25, all-age targets, into <root>/submission/
    python smh_submission.py -i ../test-output25

    # explicit selection, the optional population-group hospitalisations, quantiles too
    python smh_submission.py -i ../test-output25 --state VA MD --scenario A B \
        --age-groups --quantiles --cumulative

    # re-check a file that already exists
    python smh_submission.py --validate-only ../test-output25/submission/UVA-adaptive/*.parquet
```

## 9. Submission:  
1. Prepare the submission files: trajectories, metadata, abstract. The filenames need to satisfied the specified format, especially the "model_name" part cannot have dash "-" (so use underscore "_" if needed).
2. Create a fork from the MIDAS repository (https://github.com/midas-network/covid19-scenario-modeling-hub in case of COVID SMH). Or if a fork already exists, sync it up with the original repository.
3. Check in the submission files to the fork repository.
4. At the original MIDAS repository, create a pull request. The request will go through automatic validations and files appear in the MIDAS repository. If failed, fix or contact Lucie.