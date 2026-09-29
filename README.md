# Plant Pest Prevalence Dashboard and Simulator

This project contains two local applications for analysing and testing plant-pest surveillance methods:

- **Plant Pest Prevalence Dashboard** — applies the Change Method and Regression Method to real polygon survey data and exports a PDF report.
- **Plant Pest Simulator (beta)** — creates controlled synthetic epidemics so survey designs and analysis methods can be tested when the true prevalence is known.

Both applications run on your own computer. They open in a web browser, but the browser is only the interface; the Python application itself runs locally.

> **Important:** the Dashboard produces **decision-support estimates**, not ground-truth prevalence. The source surveillance data are polygons rather than individual host inspection records, so explicit assumptions are required to turn mapped areas into estimated surveyed hosts and infected hosts. Those assumptions are part of the result and should be read alongside the numerical estimates.

---

## Contents

1. [Quick start](#1-quick-start)
2. [What the two applications do](#2-what-the-two-applications-do)
3. [Installation](#3-installation)
4. [Recommended folder structure](#4-recommended-folder-structure)
5. [Opening the applications](#5-opening-the-applications)
6. [Dashboard: standard workflow](#6-dashboard-standard-workflow)
7. [Dashboard: data requirements](#7-dashboard-data-requirements)
8. [How the Dashboard turns polygons into prevalence](#8-how-the-dashboard-turns-polygons-into-prevalence)
9. [The two monitoring methods in plain English](#9-the-two-monitoring-methods-in-plain-english)
10. [How the sample-size calculations work](#10-how-the-sample-size-calculations-work)
11. [How to read the Results page](#11-how-to-read-the-results-page)
12. [Advanced settings explained](#12-advanced-settings-explained)
13. [Main assumptions and limitations](#13-main-assumptions-and-limitations)
14. [How flexible is the Dashboard with different polygon files?](#14-how-flexible-is-the-dashboard-with-different-polygon-files)
15. [PDF report](#15-pdf-report)
16. [Simulator guide](#16-simulator-guide)
17. [Troubleshooting](#17-troubleshooting)
18. [For developers and maintainers](#18-for-developers-and-maintainers)
19. [Scientific basis](#19-scientific-basis)

---

# 1. Quick start

If the project has already been installed on the computer:

1. Double-click **`Run Dashboard.bat`**.
2. Open the **Home** tab.
3. Expand **Data Loader**.
4. Check the three data locations.
5. Click a path box if you want to choose a different folder using the normal Windows folder browser.
6. Click **Load data**.
7. Review the **Data checks** shown inside the Data Loader.
8. Open **Data Viewer** and visually check the survey and host polygons.
9. Open **Results** and click **Run analysis**.
10. Read the plot, key metrics, interpretation, warnings, and assumptions.
11. Click **Download PDF** if a report is needed.

For most routine analyses, the default statistical settings should be left unchanged unless there is a clear reason to change them.

---

# 2. What the two applications do

## 2.1 Plant Pest Prevalence Dashboard

Use the Dashboard when you have real annual surveillance polygons and host-distribution polygons.

It can:

- load and check survey shapefiles;
- load host-distribution shapefiles;
- optionally load SPHN polygons as map context;
- display the loaded data on an interactive map;
- estimate the amount of host habitat represented by the surveys;
- estimate yearly polygon-derived prevalence;
- compare the first and last selected years using the **Change Method**;
- fit a multi-year prevalence trend using the **Regression Method**;
- compare estimated survey coverage with statistical planning references;
- show warnings and assumptions alongside the result;
- export a PDF report.

The Dashboard is designed for operational use with imperfect real-world GIS data. It performs geometry checks, attempts to repair invalid polygons, detects common year/status fields, and reports problems before analysis.

## 2.2 Plant Pest Simulator

Use the Simulator when you want to test survey designs in a controlled setting.

It can answer questions such as:

- What happens when infection is highly clustered?
- What happens when host locations are clustered?
- What happens when surveys revisit the same hosts?
- What happens when sampling is multistage rather than simple random sampling?
- How does a wrong regression curve affect the result?
- How many samples are required under different assumptions?
- How accurately do the Change Method and Regression Method recover a known true prevalence trajectory?

The Simulator is **not a disease forecast**. It creates artificial examples so that the true answer is known and the statistical methods can be tested against it.

---

# 3. Installation

## 3.1 What you need

The packaged setup is intended primarily for **Windows**.

You need:

- a Windows computer;
- internet access for the first installation;
- permission to install Python if Python is not already installed;
- the complete project folder extracted onto the computer.

Do not run the project from inside a ZIP file.

## 3.2 Install Python

Python only needs to be installed once on each computer.

1. Go to **https://www.python.org/downloads/**.
2. Download Python **3.11 or later**.
3. Run the installer.
4. On the first installer screen, tick **Add Python to PATH**.
5. Click **Install Now**.
6. Wait for installation to finish.

The **Add Python to PATH** option is important because it allows the setup script to find Python automatically.

## 3.3 Download the project from GitHub

If the project is being downloaded from GitHub:

1. Open the project repository.
2. Click **Code**.
3. Click **Download ZIP**.
4. Save the ZIP file.
5. Right-click it and choose **Extract All**.
6. Move the extracted folder to a normal writable location, for example:

```text
C:\Users\YourName\Documents\plant_pest_prevalence
```

Do not run the setup while the project is still inside the ZIP archive.

## 3.4 First-time setup

1. Open the extracted project folder.
2. Double-click **`Setup.bat`**.
3. A command window will open.
4. The setup creates a local Python environment for the project.
5. It installs the packages listed in `requirements.txt`.
6. Wait until setup reports that it has completed.

The first setup can take several minutes.

Common setup problems are:

- Python is not installed;
- Python was installed without **Add Python to PATH**;
- the computer has no internet connection;
- a firewall or proxy blocks package downloads;
- the user does not have the necessary permissions.

---

# 4. Recommended folder structure

The Dashboard now looks for its default polygon data **relative to the location of `plant_pest_dashboard.py`**. It does not rely on a developer-specific absolute path.

A typical project folder is:

```text
plant_pest_prevalence/
│
├─ plant_pest_dashboard.py
├─ plant_pest_backend.py
├─ plant_pest_simulator.py
├─ README.md
├─ requirements.txt
├─ Setup.bat
├─ Run Dashboard.bat
├─ Run Simulator.bat
│
└─ polygon_files/
   ├─ survey_data/
   │  ├─ survey_2017_2020.shp
   │  ├─ survey_2017_2020.dbf
   │  ├─ survey_2017_2020.shx
   │  ├─ survey_2017_2020.prj
   │  └─ ...
   │
   ├─ host_coverage_data/
   │  └─ ...
   │
   └─ sphn_data/
      └─ ...
```

The folders may contain one shapefile or several shapefiles.

---

# 5. Opening the applications

## 5.1 Dashboard

1. Double-click **`Run Dashboard.bat`**.
2. Keep the command window open.
3. The browser should open automatically.
4. The default local address is:

```text
http://127.0.0.1:8000
```

If the browser does not open automatically, enter that address manually.

Closing the command window stops the application.

## 5.2 Simulator

1. Double-click **`Run Simulator.bat`**.
2. Keep the command window open.
3. The default local address is:

```text
http://127.0.0.1:8001
```

The Dashboard and Simulator use different ports, so they can be open at the same time.

---

# 6. Dashboard: standard workflow

The Dashboard has three tabs:

1. **Home**
2. **Data Viewer**
3. **Results**

## 6.1 Home

The Home tab is where data are selected and checked.

### Data Loader

Expand **Data Loader** to see:

- **Survey polygons** — annual surveillance areas;
- **Host coverage** — mapped host-distribution polygons;
- **SPHN polygons (optional)** — contextual polygons displayed on the map.

The default packaged folders are filled in automatically.

### Choosing another folder

When running the Dashboard locally, clicking a path box opens the normal operating-system folder chooser.

The chooser starts from:

- the folder already shown in the box, if it exists; or
- the Dashboard application directory if the current path is invalid.

Manual path entry remains possible as a fallback.

> The folder chooser is intended for a locally run application. If the Dashboard is later hosted on a remote web server, a browser upload workflow would be needed instead because a web server cannot browse an end user's local folders directly.

### Data checks

After clicking **Load data**, the Data Loader reports:

- whether required files were found;
- whether shapefile components are present;
- whether polygons contain usable geometry;
- whether a coordinate reference system is available;
- whether invalid geometries were repaired;
- whether survey years were detected;
- whether a survey outcome/status field was detected;
- warnings and errors that need attention.

Do not ignore red errors. Warnings can sometimes be acceptable, but they should be understood before interpreting results.

## 6.2 Data Viewer

Use Data Viewer before running the analysis.

Check:

- whether survey polygons are in the expected places;
- whether the correct years appear;
- whether host coverage appears sensible;
- whether many survey areas fall outside the public host map;
- whether surveys look geographically broad or strongly concentrated;
- whether optional SPHN context is useful.

The Data Viewer is a **visual check**. It does not calculate prevalence itself.

## 6.3 Results

1. Click **Run analysis**.
2. Review the prevalence plot.
3. Review the key metrics.
4. Read the plain-English interpretation.
5. Expand **Warnings**.
6. Expand **Assumptions and limitations**.
7. Use **Advanced settings** only when a setting genuinely needs to change.
8. Use **Download PDF** to create a report.

---

# 7. Dashboard: data requirements

The Dashboard is deliberately flexible, but it is not a completely schema-free GIS application.

## 7.1 Supported vector format

The current loader supports **ESRI Shapefiles (`.shp`)**.

For each shapefile, keep the usual companion files together. The application requires:

- `.shp`
- `.dbf`
- `.shx`
- `.prj`

A `.cpg` file is optional, but recommended. If it is missing, the application warns that text encoding will be inferred.

At present the Dashboard does **not** directly load:

- GeoPackage (`.gpkg`);
- GeoJSON;
- zipped shapefiles;
- WFS/web GIS services;
- arbitrary internet URLs.

These formats could be added in future, but the current production loader is intentionally restricted to shapefiles.

## 7.2 Geometry

Survey and host layers should contain:

- `Polygon`; or
- `MultiPolygon`

features.

Non-polygon features are ignored.

The Dashboard attempts to repair invalid polygon geometries. If a geometry still cannot be made valid, loading can fail for that file.

## 7.3 Coordinate reference system

A valid CRS is required. In a shapefile this normally comes from the `.prj` file.

The application:

- converts data to WGS84 for the web map;
- converts data to **British National Grid (EPSG:27700)** for area calculations.

This means the current Dashboard is designed for Great Britain/UK operational data. It should **not** be assumed to give appropriate area calculations for unrelated countries without changing the projection strategy.

## 7.4 Survey year/date field

Survey polygons must contain a usable year or date field.

The Dashboard recognises common names such as:

- `SurveyYear`
- `SurveyYr`
- `SurvYr`
- `CreatedYear`
- `SurveyDate`
- `VisitDate`
- `InspectionDate`
- `Date`

and similar normalised variants.

If the survey uses a completely different field name and the application cannot infer a year from it, the file cannot currently contribute to the yearly analysis. This is one of the areas that may require a small code update for a new data source.

## 7.5 Survey outcome/status field

The Dashboard attempts to identify common status/outcome columns automatically.

If automatic detection fails, the file can still load. The correct column can then be selected in **Advanced settings → Survey interpretation**.

The user must specify which values mean:

- **positive** — disease/pest present;
- **negative** — disease/pest absent.

Other values are ignored.

This is useful for operational files containing labels such as pending, inconclusive, awaiting inspection, or administrative statuses.

## 7.6 Host coverage polygons

The host layer needs polygon geometry and a CRS.

No particular attribute column is mandatory if all polygons represent the host of interest.

If one file contains several host species or categories, Advanced settings can be used to choose:

- the host-category column;
- which category values to include.

## 7.7 Multiple shapefiles

A Data Loader path may point to:

- one `.shp` file; or
- a folder containing several `.shp` files.

When several files are present, the application loads all usable shapefiles in that folder and reprojects them to a common CRS if necessary.

Folder scanning is currently **one level only**. Shapefiles inside deeper subfolders are not discovered automatically.

---

# 8. How the Dashboard turns polygons into prevalence

This is the most important part of the operational interpretation.

The source files do not contain a row for every individual tree or host. They contain **survey footprints**.

The Dashboard therefore constructs a host-based estimate from spatial coverage.

## Step 1 — Build the operational host landscape

The starting host landscape comes from the public host-coverage polygons.

By default, the Dashboard also treats DEFRA survey areas outside the public host layer as evidence that host habitat exists there. Only the **uncovered portion** is added, so overlap with the public host layer is not duplicated.

This assumption exists because the public host map may be incomplete.

## Step 2 — Intersect surveys with host habitat

For each survey year, the Dashboard calculates the actual intersection between:

```text
survey footprint ∩ operational host landscape
```

Overlapping survey footprints are dissolved so the same host area is not repeatedly counted within a yearly estimate.

## Step 3 — Convert host area to estimated hosts

The Dashboard uses:

```text
estimated hosts = mapped host area × host density
```

The default host density is:

```text
2,500 hosts per km²
```

This is an assumption. The calculated host numbers are not observed tree counts unless the source data happen to correspond exactly to that density.

## Step 4 — Interpret survey status

Under the current operational interpretation:

- every estimated host inside a **negative** surveyed host footprint is treated as surveyed and uninfected;
- every estimated host inside a **positive** surveyed host footprint is treated as surveyed and infected.

Therefore:

```text
estimated yearly prevalence
=
estimated infected hosts / estimated surveyed hosts
```

A positive polygon is therefore interpreted as **all hosts in that positive surveyed host footprint being infected**.

This is a deliberate project assumption. It should not be confused with having individual host-level laboratory results.

---

# 9. The two monitoring methods in plain English

## 9.1 Change Method

The Change Method answers:

> **How much did estimated prevalence change between two selected survey years?**

It uses:

- the first selected year; and
- the last selected year.

The years in between do not affect the Change Method estimate.

Example:

```text
2017 estimated prevalence = 40%
2024 estimated prevalence = 73%

Change = 73% - 40% = +33 percentage points
```

This method is easy to interpret because both endpoints are observed directly from the survey-derived prevalence estimates.

Its main limitation is that it cannot describe what happened between those two years.

## 9.2 Regression Method

The Regression Method answers:

> **What overall prevalence trend is supported by the full sequence of survey years?**

It uses every selected survey year and fits a curve through the yearly prevalence estimates.

The standard model is **logit-linear**. Optional fractional-polynomial models are available as sensitivity analyses.

Because the Regression Method uses a fitted curve:

- its estimate of change can differ from the direct first-to-last difference;
- its estimated final prevalence can differ from the observed final-year prevalence;
- its answer depends on whether the chosen curve is a reasonable description of the real trajectory.

Neither method should be treated as automatically correct simply because it produces a number.

---

# 10. How the sample-size calculations work

This section is intentionally detailed because the Dashboard shows **three different planning quantities**. They answer different questions and should not be treated as interchangeable sample sizes.

The simplest way to remember them is:

1. **Initial prevalence precision** — “How many hosts do I need to estimate the unknown starting prevalence reasonably precisely?”
2. **Change Method** — “How many hosts do I need at each of the two compared surveys to detect the absolute change I care about?”
3. **Regression Method** — “Given the estimated starting prevalence and my target final prevalence, how much follow-up sampling do I need to detect the expected downward trend?”

The Dashboard calculates these from the values selected in **Advanced settings**. The example values from the research report are validation examples only; they are not hard-coded as universal recommendations for operational data.

## 10.1 Initial prevalence precision

Before the Regression Method can plan a trend, the starting prevalence must first be estimated with adequate precision.

The planning question is:

> **“How many hosts do I need to estimate an unknown initial prevalence reasonably precisely?”**

The calculation uses:

- **baseline confidence level**;
- **desired total confidence-interval width**;
- **baseline planning prevalence**;
- the estimated finite host population, where a finite-population adjustment is relevant.

### Baseline confidence level

This controls how confident the initial prevalence interval should be. A common choice is 0.95, meaning 95% confidence.

### Desired total confidence-interval width

This controls how narrow the initial prevalence interval should be.

For example:

```text
0.025 = a total interval width of 2.5 percentage points
```

A narrower requested interval needs a larger sample.

### Baseline planning prevalence

Before the first survey has happened, the true starting prevalence is unknown. The calculation therefore needs a reasonable planning value representing how common the pest could be.

The default value is 0.10 because this reproduces the low-prevalence planning setup used in the original worked documentation. It is **not** a universal value. If the pest could plausibly be more common, increase it.

If the observed baseline prevalence later turns out to be higher than the selected planning prevalence, the Dashboard shows a warning because the original planning assumption may have been too optimistic.

## 10.2 Change Method planning

The Change Method asks:

> **“How many hosts do I need at each of the two compared surveys to have enough power to detect an absolute prevalence change of the size I care about?”**

The requirement depends mainly on:

- observed baseline prevalence;
- the **minimum absolute change to detect**;
- significance level (`alpha`);
- power;
- planning correlation between the two prevalence estimates;
- Design Effect, if selected;
- finite population size when the calculated sample is a substantial fraction of the host population.

### The same requirement applies to both compared surveys

The Change Method sample-size expression is based on the variance of the difference between two prevalence estimates under an equal-sample-size design.

Therefore, once the baseline prevalence is known and the calculation returns a requirement such as:

```text
9,801 hosts per survey
```

the operational comparison is:

```text
First compared survey: required 9,801
Final compared survey: required 9,801
```

The Dashboard compares the historical effort in **both** selected years with this same requirement.

### Why can this only be calculated retrospectively after the first survey?

The Change Method calculation needs the baseline prevalence, but that prevalence is not known before the first survey has been conducted.

This is a genuine planning limitation. The Dashboard is able to calculate the requirement for historical data because the first survey has already happened and its estimated prevalence is available.

For a completely new programme, the initial prevalence precision calculation can be used to design the first survey. Once that survey is complete, the observed baseline prevalence can then be used to finalise the Change Method requirement for the two-survey comparison.

### Minimum change to detect

The default is:

```text
0.03
```

which means **3 percentage points**, not a 3% relative change.

For example:

- 10% → 7% is a 3 percentage-point fall;
- 40% → 37% is also a 3 percentage-point fall.

Smaller changes are harder to detect and therefore require larger samples.

### Planning correlation

The production Dashboard uses a planning correlation of zero.

This correlation is **not the same thing as the proportion of hosts resampled**. Reinspecting many of the same hosts does not automatically create an equally large correlation between the two estimated prevalences.

## 10.3 Regression Method planning

The Regression Method uses two separate stages.

### Stage 1 — estimate the baseline precisely enough

The initial prevalence survey is sized using the confidence-interval settings described in Section 10.1.

### Stage 2 — power the downward trend

Once the initial prevalence has been estimated, the Regression Method asks:

> **“Given my estimated initial prevalence and my target final prevalence, how many observations do I need in total to have enough statistical power to detect the expected slope?”**

The calculation uses:

- observed baseline prevalence;
- target/design prevalence;
- programme duration;
- scheduled follow-up times;
- significance level;
- power;
- Design Effect, if selected;
- the planned initial survey size from Stage 1.

The backend calculates the **total required follow-up sample**. The Dashboard then divides that total equally across every scheduled annual follow-up round between the selected initial and final years.

For example, if the total follow-up requirement were:

```text
1,680 hosts
```

across five follow-up years, the annual allocation would be:

```text
1,680 / 5 = 336 hosts per year
```

The initial prevalence survey is separate from this follow-up total.

### Missing historical years

The Regression Method is designed around a scheduled sequence of follow-up surveys. If the selected period is 2017–2024, the Dashboard treats 2018, 2019, 2020, 2021, 2022, 2023 and 2024 as the seven scheduled follow-up rounds.

If one of those years has no usable historical survey data, the planning comparison shows that year as having zero historical effort rather than silently removing it from the programme.

### Why can the annual Regression Method number look surprisingly small?

Because this calculation is designed to detect a **trend**, not to guarantee a precise prevalence estimate in every individual year.

A very large expected decline is statistically easy to distinguish from “no decline”. For example, planning for a fall from around 40% to 0.5% over seven years can produce a very small trend-detection allocation.

That does **not** mean:

> “A very small annual sample is enough to estimate prevalence accurately.”

It means:

> “Under the selected trend model, this amount of follow-up information is sufficient to achieve the requested power for detecting a decline as large as the one specified.”

If the starting prevalence is close to the target prevalence, the expected slope is much smaller and the required follow-up sample can increase dramatically.

## 10.4 Significance level and power

The Change Method and Regression Method use significance level and power because they are planning tests for a specified change or trend.

### Significance level (`alpha`)

Controls how strong the evidence threshold is.

For example:

```text
alpha = 0.05
```

is the usual 5% significance threshold.

Smaller values are more stringent and generally require more sampling.

### Power

Power is the probability that the planned design detects the specified change or trend when that effect is truly present.

For example:

```text
power = 0.99
```

means the design is planned for a 99% chance of detecting the specified effect under the assumptions of the calculation.

Higher power generally requires more sampling.

## 10.5 Finite population correction

If a nominal required sample becomes a substantial fraction of the estimated host population, the backend applies its finite-population correction.

This can reduce the nominal requirement because inspecting a large fraction of a finite population provides more information than the corresponding infinite-population approximation assumes.

## 10.6 Design Effect and clustered sampling

If hosts sampled within the same location are correlated, a sample of 1,000 clustered hosts does not contain as much independent information as 1,000 independent hosts.

The Dashboard can therefore use a **Design Effect** to inflate the Change Method and Regression Method follow-up requirements when a multistage/clustered reference is selected.

The polygon-based Design Effect option is only an approximation because the historical files do not contain host-by-host within-location outcomes.

## 10.7 What “planning requirement met” means

If the Dashboard says a planning requirement was met, it means approximately:

```text
estimated historical survey effort
>=
calculated statistical requirement
```

The comparison is made separately for:

- the initial prevalence precision requirement;
- each of the two Change Method surveys;
- each annual Regression Method follow-up allocation.

It does **not** mean the method is automatically trustworthy.

A numerically large enough sample cannot correct for:

- targeted surveying;
- poor geographical coverage;
- an incomplete host map;
- an inappropriate host-density assumption;
- incorrect interpretation of positive/negative polygons;
- unmodelled detection error;
- changes in survey practice through time;
- an inappropriate Regression Method curve.

For this reason the application says **planning requirement met/not met**, not **method valid/invalid**.

---

# 11. How to read the Results page

## 11.1 Prevalence plot

The plot contains different types of information:

- **yearly survey estimates** — polygon-derived prevalence for each year;
- **Change Method** — direct first-to-last comparison;
- **Regression Method** — fitted trend using all selected years.

The Change Method and Regression Method do not have to give the same final prevalence because one uses the observed endpoint directly and the other uses the fitted curve.

## 11.2 Key metrics

Typical metrics include:

### Estimated change in prevalence

Shown separately for:

- Change Method;
- Regression Method.

### Average prevalence trend

A fitted average annual change from the Regression Method.

### Estimated final prevalence

Shown separately for:

- Change Method — direct final-year estimate;
- Regression Method — fitted final-year value.

Do not describe one method's fitted endpoint as if both methods estimated exactly the same final value.

## 11.3 Interpretation

The interpretation is intended to answer questions such as:

- Did estimated prevalence increase or decrease over the selected period?
- Do both methods point in the same overall direction?
- What final prevalence does each method estimate?
- Were the relevant planning references met?

The interpretation should be read as a summary of the analysis **under the stated assumptions**, not as a statement of factual ground truth.

## 11.4 Warnings

Warnings are generated from the actual dataset and settings. They can include issues such as:

- low survey coverage relative to a planning reference;
- strongly concentrated survey footprints;
- missing or ambiguous data fields;
- a Regression Method planning target inconsistent with the observed historical direction;
- other conditions affecting interpretation.

## 11.5 Assumptions and limitations

This section summarises the main modelling and GIS assumptions used for the run.

If an assumption is not appropriate for the dataset, either change the relevant Advanced setting or treat the output as unsuitable for that decision.

---

# 12. Advanced settings explained

Advanced settings are grouped by purpose.

## 12.1 Survey interpretation

### Survey status column

The attribute field containing the survey outcome.

Use **Auto-detect** if the field is recognised correctly. Otherwise choose the correct field manually.

### Positive outcomes

Status values treated as pest/disease present.

### Negative outcomes

Status values treated as pest/disease absent.

Unticked/unselected outcomes are ignored.

---

## 12.2 Host landscape

### Host category column

Optional. Use this when one host layer contains multiple categories or species.

### Host categories to include

The values retained in the host population.

### Survey-only host areas

When enabled, portions of survey polygons outside the public host map are added as inferred host habitat.

This is based on the assumption that surveillance targeted at the host provides evidence that host habitat exists even when the public map is incomplete.

### Host density

Number of hosts assumed per square kilometre of host habitat.

Default:

```text
2,500 hosts/km²
```

Increasing the assumed density increases both the estimated total host population and the estimated number of hosts represented by survey footprints.

---

## 12.3 Planning references

### Sampling benchmark

This controls how clustering is handled in the sample-size comparison.

#### SRS reference — no Design Effect inflation

Treats the calculation as a simple-random-sampling reference.

Use this as the cleanest default benchmark when the historical survey design cannot be reconstructed reliably.

#### MSS reference — no Design Effect inflation

Uses a multistage-sampling interpretation but does not inflate sample size for within-location correlation.

#### MSS with approximate Design Effect from survey polygons

Attempts to infer an approximate clustering inflation from polygon-level data.

Because the operational dataset does not contain host-level inspection outcomes, this is an approximation. It should not be interpreted as a direct measured host-level intracluster correlation.

#### MSS with user-specified Design Effect

Uses a Design Effect supplied by the analyst.

### Hosts per location visit for MSS

Assumed number of hosts inspected during one sampled-location visit.

The research simulation commonly used five hosts per visit.

### Fixed Design Effect

Manual inflation factor used only when the user-specified Design Effect option is selected.

A Design Effect of:

```text
1.0
```

means no inflation.

A value of:

```text
2.0
```

means approximately twice as many observations are required to compensate for clustering.

### Significance level (`alpha`)

Controls how strong the evidence threshold is for the planning test.

Default:

```text
0.05
```

Smaller alpha values are more stringent and generally require more samples.

### Power

Power is the probability that the planned design detects the specified change/trend when that effect really exists.

Dashboard default:

```text
0.99
```

Higher power generally requires more samples.

---

## 12.4 Initial prevalence planning settings

### Baseline confidence level

Confidence level used to size the first survey for estimating the unknown starting prevalence.

### Desired total confidence-interval width

Maximum total width requested for the initial prevalence confidence interval.

Example:

```text
0.025 = 2.5 percentage points total width
```

### Baseline planning prevalence

A conservative prevalence assumption used before the initial survey is available. This should be changed when 10% is not a defensible upper planning value for the pest and target population.

---

## 12.5 Change Method settings


### First survey year

Start year for the direct comparison.

### Last survey year

End year for the direct comparison.

### Minimum change to detect

The absolute prevalence difference the planning calculation is designed to detect.

Example:

```text
0.03 = 3 percentage points
```

Smaller values require larger samples.

The production Dashboard uses a planning correlation of zero unless the backend/design is deliberately changed.

---

## 12.6 Regression Method settings

### Initial survey year

First year used in the fitted trend.

### Final survey year

Last year used in the fitted trend.

The planning calculation uses the full scheduled annual sequence between the selected initial and final years. Missing historical survey years are retained as zero historical effort in the planning comparison.

### Regression model

Available forms include:

- **Logit-linear** — primary/default model;
- **Fractional polynomial 2** — sensitivity analysis;
- **Fractional polynomial 3** — sensitivity analysis.

More flexible is not automatically better. A model can fit noise as well as signal.

### Target prevalence

The prevalence that defines the downward trend the Regression Method is planned to detect.

Default:

```text
0.005 = 0.5%
```

This is a **planning target**, not a statement that the historical data actually followed that path.

If baseline prevalence is already at or below the target, a negative target slope cannot be defined and Regression Method trend planning is reported as not applicable.

---

# 13. Main assumptions and limitations

The most important assumptions are listed here in plain language.

## 13.1 Survey footprint = surveyed hosts

Every estimated host lying inside the usable surveyed host footprint is treated as having been surveyed.

## 13.2 Positive polygon = all hosts infected

Every estimated host in the host-covered part of a polygon classified as positive is treated as infected.

A negative polygon contributes surveyed hosts but no infected hosts.

## 13.3 Host counts are estimated, not counted

Host numbers come from:

```text
mapped area × assumed host density
```

They are not observed host-level counts.

## 13.4 Public host coverage may be incomplete

Survey areas outside the public host map can be added as inferred host habitat.

This is useful when the public map is incomplete, but it assumes the surveyed footprint represents host-containing area.

## 13.5 Overlap is dissolved

Survey/host intersections are used and overlapping survey footprints are dissolved so the same area is not counted repeatedly within a year.

## 13.6 Only selected outcomes enter the prevalence calculation

Positive and negative categories are selected explicitly. Other statuses are ignored.

## 13.7 No correction for imperfect detection

The recorded polygon classification is taken at face value. The operational Dashboard does not estimate diagnostic/surveillance sensitivity from the polygon files.

## 13.8 Historical surveys may be targeted

If survey teams preferentially visited high-risk or previously affected areas, the calculated prevalence can be systematically different from prevalence in the wider host population.

A larger sample size does not remove this bias.

## 13.9 Surveys need to be comparable through time

Changes in:

- targeting;
- polygon construction;
- recording practice;
- host mapping;
- surveillance intensity;

can appear as prevalence change even if underlying biology did not change by the same amount.

## 13.10 Regression results are model-dependent

The Regression Method assumes a temporal curve. If the real trajectory has another shape, the fitted change, slope, or final prevalence can be biased.

## 13.11 Confidence intervals do not include every source of uncertainty

Model-based uncertainty does not automatically propagate uncertainty in:

- host density;
- public host-map completeness;
- polygon interpretation;
- survey status classification;
- preferential site selection.

## 13.12 Association is not causation

An estimated increase or decrease in prevalence does not, by itself, prove that management measures caused the change.

## 13.13 SPHN polygons are contextual

SPHN polygons are displayed for map context. They do not directly enter the prevalence calculation unless the underlying survey polygons themselves contain the relevant survey-status information.

---

# 14. How flexible is the Dashboard with different polygon files?

## 14.1 What it handles well

The Dashboard is reasonably adaptable to alternative UK polygon datasets when they follow a conventional surveillance structure.

It can usually handle:

- different shapefile names;
- one or several shapefiles in a selected folder;
- different years in different files;
- different CRSs, provided each file has a valid CRS;
- extra attribute columns;
- alternative status fields, because the status column can be chosen manually;
- host layers with or without category fields;
- incomplete public host coverage, using the survey-only host-area option.

## 14.2 Situations that can break or block analysis

Analysis may fail or become inappropriate when:

- the selected path does not exist;
- no `.shp` files are present;
- required shapefile components are missing;
- the layer has no CRS;
- there are no usable polygon geometries;
- invalid geometry cannot be repaired;
- survey years cannot be detected;
- fewer than two usable survey years are present;
- no survey outcome field can be selected;
- no outcomes are classified as positive or negative;
- the scientific meaning of a polygon does not match the Dashboard's assumptions;
- the data are outside Great Britain but are still forced through British National Grid area calculations.

## 14.3 What to change for a new organisational dataset

For a new dataset, first try it without changing the code.

If loading fails, check in this order:

1. Is it an ESRI shapefile?
2. Are `.shp`, `.dbf`, `.shx` and `.prj` present together?
3. Are the geometries Polygon/MultiPolygon?
4. Is the year stored in a recognisable year/date column?
5. Can the survey outcome field be selected in Advanced settings?
6. Does the host layer need a category filter?
7. Are the polygons genuinely comparable to the full-coverage and positive-polygon assumptions?

If the only problem is an unusual year-field name, the list of recognised year/date fields is centralised near the top of `plant_pest_dashboard.py` and can be extended easily.

---

# 15. PDF report

The PDF is intended as a concise decision-support record of the analysis.

It includes three broad sections.

## 15.1 Result summary

Typically includes:

- prevalence through time;
- Change Method and Regression Method change estimates;
- method-specific final prevalence estimates;
- fitted average trend;
- short interpretation.

## 15.2 Survey evidence

Typically includes:

- yearly survey summaries;
- observed prevalence estimates;
- host coverage represented;
- spatial coverage indicators;
- important data-specific warnings.

## 15.3 Planning and analysis record

Typically includes:

- initial prevalence precision requirement;
- Change Method per-survey requirement at both compared years;
- Regression Method total follow-up requirement and per-round allocation;
- main analysis settings;
- concise assumptions and limitations.

The planning section should not be read as proof that a historical survey was representative or unbiased.

---

# 16. Simulator guide

The Simulator follows the same broad logic as the research study but uses synthetic data.

## 16.1 Recommended first run

1. Open **Run Simulator.bat**.
2. Create a synthetic host landscape.
3. Keep the default prevalence curve.
4. Generate the infection landscape.
5. Estimate sampling effort.
6. Simulate surveys.
7. Fit the Change Method and Regression Method.
8. Compare the estimates with the known true prevalence.

Start with one simulation so that the workflow is easy to inspect. Use repeated simulations only after the scenario has been configured correctly.

## 16.2 Host Landscape

Creates the population of hosts.

Options include:

- real host polygons;
- synthetic clusters;
- different host-clustering levels;
- host density;
- total simulated area;
- random seed.

## 16.3 True Prevalence

Defines the true prevalence trajectory that the methods are trying to recover.

Key settings include:

- number of survey rounds;
- initial prevalence;
- final prevalence;
- true trajectory shape;
- prevalence-level shift.

## 16.4 Infection Landscape

Places infection on the host landscape while maintaining the specified population prevalence.

Options range from random infection to increasingly concentrated hotspots.

## 16.5 Sampling Effort

Calculates method-specific planning sample sizes.

Important settings include:

- significance level;
- power;
- Change Method minimum detectable change;
- Regression Method target prevalence;
- clustering inflation/Design Effect.

## 16.6 Survey Design

Controls how synthetic surveys are carried out.

Options include:

- simple random sampling;
- multistage sampling;
- targeted sampling;
- host resampling/retention;
- detection sensitivity;
- hosts per sampled location visit.

Host retention changes which hosts are revisited. It does **not** automatically equal the Change Method planning correlation.

## 16.7 Results

The Simulator compares estimated results with the known truth.

It reports information such as:

- prevalence estimates;
- change estimates;
- Regression Method fitted trend;
- final prevalence error;
- requested versus achieved survey effort;
- survey overlap;
- Design Effect/ICC diagnostics;
- regression convergence and validity checks.

## 16.8 Scenario Comparison

Scenario Comparison is useful for controlled “what if?” experiments.

Good practice is to keep Scenario A fixed and change one important feature in Scenario B, for example:

- random vs clustered infection;
- SRS vs multistage sampling;
- low vs high host retention;
- no Design Effect inflation vs inflation;
- logit-linear vs fractional-polynomial fitting.

With repeated simulations:

- errors near zero are better;
- positive error means overestimation;
- negative error means underestimation;
- wider distributions mean less precise/reliable estimates.

---

# 17. Troubleshooting

## Setup says Python is not recognised

1. Install Python 3.11 or later.
2. Ensure **Add Python to PATH** is selected during installation.
3. Run `Setup.bat` again.

## Setup fails while installing packages

Check:

- internet access;
- firewall/proxy restrictions;
- user installation permissions.

## Dashboard does not open

Check that:

- setup has completed;
- the command window is still open;
- no error is shown in the command window;
- `http://127.0.0.1:8000` is being used.

## Simulator does not open

Check:

- the command window is still open;
- `http://127.0.0.1:8001` is being used.

## Folder chooser does not open

The native chooser depends on the local Python process being able to open a desktop window.

If it does not work:

- type or paste the folder path manually;
- make sure the application is being run locally rather than on a remote/server environment.

## Shapefiles do not load

Check:

- the path points to the correct folder;
- `.shp`, `.dbf`, `.shx`, and `.prj` are together;
- the data contain polygons;
- the CRS is defined;
- a usable year/date field exists for survey data.

## Map loads but polygons are missing

Check:

- the correct year is selected;
- the relevant layer is ticked in Data Viewer;
- the layer passed Data Checks;
- its coordinates/CRS are correct.

## Results cannot run

Common causes include:

- fewer than two usable survey years;
- no positive/negative statuses selected;
- no host coverage available;
- invalid/missing geometry or CRS;
- selected analysis years not present in the data.

## Regression Method planning is unavailable

If estimated baseline prevalence is already at or below the selected target prevalence, there is no negative target slope to power. The trend-planning calculation is therefore not applicable.

## PDF does not generate

1. Run the analysis first.
2. Check the command window for errors.
3. Re-run setup if reporting dependencies are missing.

## Simulator is slow

- start with one run;
- use fewer synthetic clusters while testing settings;
- only increase Monte Carlo repetitions once the scenario is configured correctly.

---

# 18. For developers and maintainers

The code is deliberately separated so the scientific calculations are not duplicated across applications.

## 18.1 Main files

### `plant_pest_dashboard.py`

Responsible for:

- data loading;
- shapefile validation;
- GIS overlay/intersection logic;
- operational polygon interpretation;
- UI;
- report generation;
- calling the statistical backend.

### `plant_pest_backend.py`

Responsible for:

- baseline prevalence sample-size calculation;
- Change Method planning;
- Regression Method planning;
- Design Effect calculations;
- simulation landscape and survey logic;
- regression fitting and diagnostics;
- reference-value validation.

The backend is the scientific implementation shared by the Dashboard and Simulator. Avoid copying formulas into the UI files.

### `plant_pest_simulator.py`

Responsible for:

- interactive simulation workflow;
- scenario controls;
- calling the shared backend;
- displaying simulated results.

## 18.2 Basic code checks after changes

Compile all main files:

```bash
python -m py_compile plant_pest_backend.py plant_pest_dashboard.py plant_pest_simulator.py
```

Run the backend reference validations:

```bash
python -c "from plant_pest_backend import validate_efsa_reference_examples; print(validate_efsa_reference_examples())"
```

The reference checks include the established planning benchmarks such as:

- initial prevalence reference around 2,292;
- five-year Regression Method follow-up total around 1,680;
- about 336 per follow-up year in the published reference example;
- Change Method sample-size examples under different planning correlations.

## 18.3 Rules for future changes

When changing the software:

1. Keep statistical formulas in `plant_pest_backend.py`.
2. Keep GIS assumptions explicit in `plant_pest_dashboard.py`.
3. Do not silently reinterpret unknown survey fields.
4. Add new recognised field names centrally rather than scattering special cases through the code.
5. Preserve data checks and warnings when adding more flexible loading.
6. Re-run the reference validations after changing any planning calculation.
7. Update this README whenever the user-visible workflow or a scientific assumption changes.

A separate maintainer guide may contain more implementation detail, but this README should remain the first place a new user or developer starts.

---

# 19. Scientific basis

The two monitoring approaches implemented in this project come from EFSA guidance on monitoring pest prevalence through time and were evaluated in the accompanying research study:

**Estimating Change in Plant Pest Prevalence: A Simulation Study of EFSA Monitoring Methods**

The study evaluated the methods under conditions including:

- different prevalence-change magnitudes;
- repeated host sampling/retention;
- spatial host clustering;
- spatial infection clustering;
- multistage sampling;
- Design Effect inflation;
- regression-model misspecification;
- different prevalence levels.

The research study used a shared baseline survey for method comparison. The operational Dashboard now keeps that research convention separate from operational planning: the initial precision survey is calculated from the user-selected confidence settings, the Change Method requirement is applied equally to both compared surveys, and the Regression Method follow-up total is calculated and divided across scheduled annual rounds.

The important practical conclusion is not that one method is universally better. The methods answer different questions:

- use the **Change Method** when the main quantity of interest is the direct difference between two survey occasions;
- use the **Regression Method** when the programme needs evidence about the trajectory through time and accepts the additional model assumptions that come with fitting a trend.

In all cases, statistical sample size is only one part of survey quality. Representativeness, geographical coverage, host-map quality, status interpretation, clustering, detection sensitivity, and consistency of surveillance practice remain important.

---

## Final reminder

Before using a Dashboard result in a decision, check four things:

1. **Did the data load correctly?**
2. **Does the Data Viewer show a sensible survey footprint?**
3. **Were the relevant planning references met, and what do those references actually measure?**
4. **Are the assumptions reasonable for this dataset and decision?**

If any of those answers is uncertain, the result should be treated as uncertain too.
