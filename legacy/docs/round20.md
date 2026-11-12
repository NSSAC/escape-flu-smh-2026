# Covid19 Round 20

* https://github.com/midas-network/covid19-scenario-modeling-hub

* projection period:
    * two years
    * Sunday, June 8, 2025 -- Saturday, June 5, 2027
        * 2025-06-08 -- 2027-06-05
* vaccine effectiveness:
    * Teams should use VE against COVID-19 hospitalization = 55%
        at the start of the vaccination campaign:
        * August 13, 2025 for the 2025-26
        * August 14, 2026 for the 2026-27
    * stated hospitalization reduction includes
        the combined effects of protection against infection
        and protection against hospitalization
    * VE against infection in the range 35-57%
* immune escape:
    * immune escape against infection and severe disease is at teams' discretion
    * we suggest that it should be bounded by 20-50% per year.
    * Teams should note that the impact of immune escape is separate from the impact of waning immunity
* waning immunity:
    * we suggest a 3-10 month waning time, with 40-60% reduction against baseline protection levels in waned state.
    * Waning of vaccine-induced immunity against severe disease
        remains at teams discretion
        (but if present should wane at slower timescale than against infection).
* vaccine frequency:
    * Only high-risk individuals can benefit
    from the additional opportunity modeled in scenarios C and E,
    with only (half of the) individuals already vaccinated
    during the main campaign receiving a second dose.
    * While we expect that the risk of getting vaccinated
    in the Spring campaign is correlated to the propensity
    to get vaccinated in the Fall campaign
    * Teams can choose to model correlated outcomes or not.
    * Details of the second dose implementation should be provided in the Appendix.

* Hospitalization data:
    * https://github.com/midas-network/covid19-scenario-modeling-hub/blob/main/target-data/time-series.csv

* Vaccination curves:
    * https://github.com/midas-network/covid19-scenario-modeling-hub/blob/main/auxiliary-data/vaccination-coverage/COVID_RD19_Vaccination_curves.csv

* Synthetic population:
    * /project/bii_nssac/EpiHiperSynPop/v2.4.0/from-alt-project

* U.S. Chronic Disease Indicators
    * https://data.cdc.gov/Chronic-Disease-Indicators/U-S-Chronic-Disease-Indicators/hksd-2xuw/about_data

* Preliminary U.S. COVID-19 Burden Estimates
    * https://data.cdc.gov/Public-Health-Surveillance/Preliminary-U-S-COVID-19-Burden-Estimates/ahrf-yqdt/about_data
