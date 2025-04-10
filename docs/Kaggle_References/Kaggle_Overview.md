Stanford RNA 3D Folding
Solve RNA structure prediction, one of biology's remaining grand challenges


Stanford RNA 3D Folding

Submit Prediction
Overview
If you sat down to complete a puzzle without knowing what it should look like, you’d have to rely on patterns and logic to piece it together. In the same way, predicting Ribonucleic acid (RNA)’s 3D structure involves using only its sequence to figure out how it folds into the structures that define its function.

In this competition, you’ll develop machine learning models to predict an RNA molecule’s 3D structure from its sequence. The goal is to improve our understanding of biological processes and drive new advancements in medicine and biotechnology.

Start

a month ago
Close
2 months to go
Merger & Entry
Description
RNA is vital to life’s most essential processes, but despite its significance, predicting its 3D structure is still difficult. Deep learning breakthroughs like AlphaFold have transformed protein structure prediction, but progress with RNA has been much slower due to limited data and evaluation methods.

This competition builds on recent advances, like the deep learning foundation model RibonanzaNet, which emerged from a prior Kaggle competition. Now, you’ll take on the next challenge—predicting RNA’s full 3D structure.

Your work could push RNA-based medicine forward, making treatments like cancer immunotherapies and CRISPR gene editing more accessible and effective. More fundamentally, your work may be the key step in illuminating the folds and functions of natural RNA molecules, which have been called the 'dark matter of biology'.

This competition is made possible through a worldwide collaborative effort including the organizers, experimental RNA structural biologists, and predictors of the CASP16 and RNA-Puzzles competitions; Howard Hughes Medical Institute; the Institute of Protein Design; and Stanford University School of Medicine.

Evaluation
Submissions are scored using TM-score ("template modeling" score), which goes from 0.0 to 1.0 (higher is better):

TM-score=max⎛⎝⎜⎜1Lref∑i=1Lalign11+(did0)2⎞⎠⎟⎟

where:

Lref is the number of residues solved in the experimental reference structure ("ground truth").

Lalign is the number of aligned residues.

di is the distance between the ith pair of aligned residues, in Angstroms.

d0 is a distance scaling factor in Angstroms, defined as:

d0=0.6(Lref−0.5)1/2−2.5

for Lref ≥ 30; and d0 = 0.3, 0.4, 0.5, 0.6, or 0.7 for Lref <12, 12-15, 16-19, 20-23, or 24-29, respectively.

The rotation and translation of predicted structures to align with experimental reference structures are carried out by US-align. To match default settings, as used in the CASP competitions, the alignment will be sequence-independent.

For each target RNA sequence, you will submit 5 predictions and your final score will be the average of best-of-5 TM-scores of all targets. For a few targets, multiple slightly different structures have been captured experimentally; your predictions' scores will be based on the best TM-score compared to each of these reference structures.

Submission File
For each sequence in the test set, you can predict five structures. Your notebook should look for a file test_sequences.csv and output submission.csv. This file should contain x, y, z coordinates of the C1' atom in each residue across your predicted structures 1 to 5:

ID,resname,resid,x_1,y_1,z_1,... x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,... -7.301,9.023,8.932
R1107_2,G,1,-8.02,11.014,14.606,... -7.953,10.02,12.127
etc.
You must submit five sets of coordinates.

Timeline
February 27, 2025 - Start Date.
April 23, 2025 - Public leaderboard refresh & Early Sharing Prizes
May 22, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.
May 22, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.
May 29, 2025 - Final Submission Deadline.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Future Data Evaluation Timeline:
After the final submission deadline there will be periodic updates to the leaderboard to reflect up to 40 new RNA (sequences) generated after the competition has ended. New data updates that will be run against selected notebooks.

September 24, 2025 - Competition End Date - This date is subject to change based upon the availability of new sequences. Watch the forum after the competition end for updates.
Prizes
Leaderboard Prizes
1st Place - $ 45,000
2nd Place - $ 15,000
3rd Place - $ 10,000
Early Sharing Prizes
Participants of this competition are encouraged to make publicly available their notebooks through the competition. There will be a refresh of the public leaderboard 2 months after competition start. At that time, $2,500 will be awarded to the first two teams to publish a public notebook scoring above the VFOLD_human_expert score on the leaderboard. A discussion post will detail timing of the refresh.

To be eligible for the Early Sharing Prize, you will need to:

1) Publish a public notebook scoring above the benchmark score on the leaderboard after the data refresh (first two notebooks that meet this criteria will be evaluated).

2) Out of all participants or Teams who have submitted notebooks scoring above the benchmark score, be the first two to make your notebooks public. The public notebook needs to adhere to the same requirements and restrictions regarding licensing, reproducibility, and documentation to which the winning Submission is subject (see Competition Rules).

3) Keep the notebooks and any datasets they use publicly available until the final Progress Prizes are awarded to the winning Teams at the end of the competition. Submissions should only make use of information publicly available before the temporal_cutoff dates provided with test sequences.

The Competition Sponsor will, after the data refresh, assess all Submissions that are eligible for the Early Sharing Prize in the order in which Submissions were made. If it is discovered that such a Submissions that scored more than the benchmark score has no or incomplete documentation, incompatible licensing, or is in any other way incompatible with the rules to which the winning Submission is subject, it will not be considered towards the Early Sharing Prize and the next Submissions will be assessed.

Paper Authorship
Top performing participants on the Public Leaderboard rankings at the final submission deadline will be invited to contribute their code and model descriptions to a scientific paper summarizing the competition's scientific outcome.

Code Requirements


This is a Code Competition
Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 8 hours run-time
GPU Notebook <= 8 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Submission file must be named submission.csv
Submission runtimes have been slightly obfuscated. If you repeat the exact same submission you will see up to 5 minutes of variance in the time before you receive your score.
Future Data Evaluation Phase
The run-time limits for both CPU and GPU notebooks will be extended to during the future data evaluation period proportional to the number of future samples. You must ensure your submission completes within that time. The extra runtime will enable us to use a substantially larger test set as the basis for ranking submissions on the final private leaderboard.

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

Additional Resources
What's the state-of-the-art in RNA 3D structure prediction?
2024 CASP16 challenge, including presentations from this competition's hosts
Latest results from RNA-Puzzles, including predictions from this competition's hosts

The RibonanzaNet foundation model
Ribonanza: deep learning of RNA structure through dual crowdsourcing

Stanford Ribonanza RNA Folding Kaggle challenge

How to think about RNA structure
A perspective from domain experts

Cartoon explanation (XKCD)
