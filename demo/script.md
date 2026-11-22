# Cricket X-Ray: demonstration video script

Video: `demo/Cricket_XRay_Demo.mp4` · 06:29.1 · 1920×1080, 30 fps, H.264 + AAC, soft English subtitles
Narration voice: `en-IN-PrabhatNeural` (edge-tts). Rebuild with `.venv/bin/python demo/make_video.py`.

Timestamps are generated from the final timeline, so they match the video exactly.

| # | Scene | Start | End | Length |
|---|---|---|---|---|
| 1 | Title | 00:00.0 | 00:13.9 | 13.9 s |
| 2 | Problem | 00:13.9 | 00:43.4 | 29.5 s |
| 3 | Data | 00:43.4 | 01:10.4 | 27.0 s |
| 4 | Pipeline | 01:10.4 | 01:48.0 | 37.6 s |
| 5 | Live demo · Overview | 01:48.0 | 02:13.9 | 25.9 s |
| 6 | Live demo · Data | 02:13.9 | 02:29.8 | 15.9 s |
| 7 | Live demo · Features | 02:29.8 | 02:53.0 | 23.2 s |
| 8 | Live demo · Model evaluation | 02:53.0 | 03:19.8 | 26.8 s |
| 9 | Live demo · Clusters | 03:19.8 | 04:01.1 | 41.3 s |
| 10 | Live demo · Player explorer | 04:01.1 | 04:28.3 | 27.3 s |
| 11 | Live demo · Changing K | 04:28.3 | 04:52.3 | 24.0 s |
| 12 | Live demo · Methodology | 04:52.3 | 05:05.2 | 12.8 s |
| 13 | Tests | 05:05.2 | 05:26.8 | 21.6 s |
| 14 | Validation | 05:26.8 | 05:52.6 | 25.8 s |
| 15 | Conclusion | 05:52.6 | 06:29.1 | 36.5 s |

## 00:00.0 · Title

*On screen:* card `title`

- `[00:00.6]` Welcome to Cricket X-Ray: discovering cricket player performance profiles with unsupervised machine learning.
- `[00:07.8]` A Milestone 3 project by Arjav Jain, Nishkarsh Sisodia and Drishti Chauhan.

## 00:13.9 · Problem

*On screen:* card `problem`

- `[00:14.5]` Cricket players are usually described by informal roles: batter, bowler, all-rounder and wicket-keeper.
- `[00:21.7]` But roles overlap: many batters also bowl, and many keepers are leading batters.
- `[00:28.1]` There is no reliable role label to learn from, so this is an unsupervised problem.
- `[00:34.0]` Our goal: group T20 players by how they contribute, using career statistics alone, and explain each group in cricket terms.

## 00:43.4 · Data

*On screen:* card `data`

- `[00:44.0]` We use three T20 international career tables: batting, bowling and fielding.
- `[00:49.7]` Each holds the same 2,006 players, merged on the full name with its country tag.
- `[00:55.7]` A dash means missing, not zero: a player who never bowled has no economy rate.
- `[01:01.8]` And overs are not decimals: 10.5 overs is 10 overs and 5 balls, or 65 balls.

## 01:10.4 · Pipeline

*On screen:* card `pipeline`

- `[01:11.0]` Here is the pipeline.
- `[01:12.8]` We load and validate the files into one table of 2,006 players and 44 columns.
- `[01:19.7]` We keep the 1,105 players with at least five matches, and engineer twelve per-innings and per-match rates.
- `[01:28.5]` For the model, we clip each feature to its 1st to 99th percentile, fill undefined rates with the median, add a has-bowled flag, and standardize.
- `[01:38.9]` K-Means groups the players.
- `[01:41.0]` PCA maps them in two dimensions.
- `[01:43.9]` And simple rules give each cluster a readable label.

## 01:48.0 · Live demo · Overview

*On screen:* live Streamlit dashboard, driven by Playwright

- `[01:48.6]` Now, the live Streamlit dashboard.
- `[01:51.4]` The sidebar sets the minimum matches, 5, and the number of clusters K, 4.
- `[01:57.1]` The Overview shows 2,006 source players, 1,105 modelled players, 9 model features, and a silhouette score of 0.269.
- `[02:08.8]` Below are a PCA preview and the size of each cluster.

## 02:13.9 · Live demo · Data

*On screen:* live Streamlit dashboard, driven by Playwright

- `[02:13.9]` The Data tab checks the inputs.
- `[02:16.3]` Three source files, 2,006 rows each, no duplicates, and every player appears in all three.
- `[02:24.3]` Here are the missing values per column; a dash always stays missing.

## 02:29.8 · Live demo · Features

*On screen:* live Streamlit dashboard, driven by Playwright

- `[02:29.8]` In Features, the filter removes 901 players with fewer than five matches, leaving 1,105.
- `[02:38.8]` We engineer twelve rates in three groups: batting, bowling and fielding.
- `[02:44.0]` The definitions table gives each formula.
- `[02:47.0]` Eight of these rates, plus the has-bowled flag, form the nine model inputs.

## 02:53.0 · Live demo · Model evaluation

*On screen:* live Streamlit dashboard, driven by Playwright

- `[02:53.0]` Model evaluation fits K-Means for every K from 2 to 8.
- `[02:57.6]` The elbow bends most sharply at K = 3, while the silhouette peaks at K = 4, at 0.269.
- `[03:06.8]` We chose K = 4: the best silhouette, and four clusters that are easy to interpret.
- `[03:13.6]` A score this modest means weak structure, as expected, because roles overlap.

## 03:19.8 · Live demo · Clusters

*On screen:* live Streamlit dashboard, driven by Playwright

- `[03:19.8]` The Clusters tab shows the result for K = 4.
- `[03:23.6]` The PCA map keeps 51.6% of the variance; each cluster has its own colour and marker.
- `[03:31.4]` The four profiles: Bowling-Oriented, 463 players; Batting-Oriented Also Bowls, 320; Batting-Oriented Rarely Bowls, 216; and Wicket-Keeping, 106.
- `[03:46.1]` The profile table gives the real median statistics.
- `[03:49.6]` The heatmap shows relative strengths, where the keepers stand out at +3.0 standard deviations.
- `[03:57.1]` And the results can be downloaded as CSV.

## 04:01.1 · Live demo · Player explorer

*On screen:* live Streamlit dashboard, driven by Playwright

- `[04:01.1]` The Player explorer looks at one player at a time.
- `[04:04.9]` Let's search for V Kohli of India.
- `[04:07.7]` He falls in Cluster 2, Batting-Oriented, Also Bowls.
- `[04:11.9]` This panel shows that Cluster 2 is his nearest centre, and compares his standardized values with it.
- `[04:19.1]` Below are his source statistics, exactly as in the CSV files.
- `[04:24.4]` And here are his five most similar players.

## 04:28.3 · Live demo · Changing K

*On screen:* live Streamlit dashboard, driven by Playwright

- `[04:28.3]` Every tab follows the sidebar.
- `[04:30.7]` Let's change K from 4 to 3.
- `[04:33.4]` The model retrains at once: the metrics, the PCA map and the cluster sizes all update, and the silhouette is now 0.265.
- `[04:43.4]` Clusters now shows three profiles.
- `[04:47.6]` Back to K = 4, and the four profiles return.

## 04:52.3 · Live demo · Methodology

*On screen:* live Streamlit dashboard, driven by Playwright

- `[04:52.3]` Finally, the Methodology tab explains every step in plain words, from the problem to the cluster labels.
- `[04:59.5]` It ends with the model inputs, the limitations and future work.

## 05:05.2 · Tests

*On screen:* card `tests`

- `[05:05.8]` The project has 80 automated tests, written with unittest and Streamlit's AppTest.
- `[05:11.7]` They cover data loading, preprocessing, modeling and the dashboard.
- `[05:16.5]` With the real data present, all 80 pass.
- `[05:19.8]` Without the data, as on GitHub Actions, 52 pass and 28 skip by design.

## 05:26.8 · Validation

*On screen:* card `validation`

- `[05:27.4]` Real players have no true labels, so we also built a synthetic league of 60 players, 15 of each type.
- `[05:35.0]` With K = 4, K-Means recovered all four types with 100% purity, and every label was correct.
- `[05:43.4]` Its silhouette of 0.583 shows that the modest real score comes from overlapping roles, not from the pipeline.

## 05:52.6 · Conclusion

*On screen:* card `conclusion`

- `[05:53.2]` To conclude: with no labels, Cricket X-Ray found four interpretable profiles among 1,105 T20 players.
- `[06:02.2]` The 0.269 silhouette shows that roles form a continuum, and the synthetic test shows that the pipeline works.
- `[06:10.6]` Limits: career totals ignore batting position, era and form, and K-Means assumes compact clusters.
- `[06:18.2]` Next, we would add ODI and Test data, try Gaussian mixtures, and use ball-by-ball data.
- `[06:25.4]` The code is on GitHub. Thank you!

## Sources

All figures come from `Cricket_XRay_Project_Report.pdf` and the repository; `make_video.py` re-checks them against a live pipeline run before building. The test figures (80 run, 0 skipped, 0 failed) come from the test run captured during the build.
