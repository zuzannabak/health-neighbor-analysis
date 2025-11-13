# Healthy-Lifestyle Adoption on Reddit

A network science project analyzing how health-related behaviors spread across Reddit communities.

## Project Summary

- Built subreddit–subreddit interaction graphs from the [Reddit Hyperlinks Network (SNAP)](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) dataset (2014–2017).
- Labeled “healthy-lifestyle” subreddits based on LIWC Body & Health indicators.
- Measured how connections to healthy communities predict later adoption of health-related topics.
- Found that connected subreddits were **2–3× more likely** to become health-focused.

## Results Summary

| Period (T1 → T2) | P(with neighbor) | P(no neighbor) | p-value |
|------------------:|-----------------:|----------------:|---------:|
| 2014H1–2014H2 | 0.0998 | 0.0398 | 2.21e−08 |
| 2014H2–2015H1 | 0.0980 | 0.0435 | 4.53e−09 |
| 2015H1–2015H2 | 0.0772 | 0.0355 | 1.04e−07 |
| 2015H2–2016H1 | 0.0803 | 0.0398 | 6.18e−08 |
| 2016H1–2016H2 | 0.1072 | 0.0356 | 4.45e−22 |
| 2016H2–2017H1 | 0.1060 | 0.0282 | 5.20e−30 |

> Communities connected to health-related subreddits were consistently 2–3× more likely to adopt health topics.  
> All results were statistically significant (p < 0.001).


## Tech Stack

- Python  
- pandas, numpy, networkx, matplotlib, statsmodels  

## How to Run

1. Download `soc-redditHyperlinks-body.tsv` from [SNAP](https://snap.stanford.edu/data/soc-RedditHyperlinks.html).  
   Place it in a local `data/` folder.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis:
   ```bash
   python health_neighbor_analysis.py
   ```

## Additional Materials

* `project_report.pdf` – full written report (methods, results, discussion).

* `project_presentation.pdf` – slide deck used for the project presentation.

## Author
Zuzanna Bąk
M.S. Computational Data Science, Temple University
zuzanna.bak@temple.edu
