# kaggle-renthop-2017
https://www.kaggle.com/ericdoi

A public example Kaggle submission for Two Sigma Connect: Rental Listing Inquiries, 2017 [(Kaggle competition page)](https://kaggle.com/c/two-sigma-connect-rental-listing-inquiries)

Some slides about my experience working on this competition: [(pdf)](https://github.com/ericdoi/kaggle-renthop-2017/blob/master/2017-07-01_Kaggle_RentHop.pdf)

## Description
This competition was notable for its rich domain-specific features and manageable data size,
aside from a 80GB image dataset (however, the consensus was that image processing yielded little lift over the data in the standard
numeric/text data).

The structure of this repository is modeled after Jeong-Yoon Lee's Kaggler Template (https://github.com/jeongyoonlee/kaggler-template).
It is organized to make model runs reproducible and facilitate ensembling of out-of-fold model predictions.

Makefiles are used to run the modeling pipeline (data prep, model training, prediction, and ensembling).
For overview and motivation, see http://kaggler.com/kagglers-toolbox-setup/.

For modeling, XGB was the main model, ensembled with Logistic Regression.

## Result
Private Leaderboard:  0.51150

Rank 226 / 2488 (91st percentile)

This was based on the following ensemble:
* XGB with f6a (feature.f6a.makefile) features
* XGB with f6 (feature.f6.makefile) + img1 (feature.img1.makefile, after running generate_image_feats.py) features
* LR with n6 (feature.n6.makefile) + pq (generate_price_quantiles.py) features
* LR with img1 features
* LR with n6a (feature.n6a.makefile) + img1 features

## Acknowledgements
Thanks to:
- [Jeongyoonlee](https://www.kaggle.com/jeongyoonlee):  For the Kaggler toolbox and template
- [plantsgo](https://www.kaggle.com/plantsgo):  For generous sharing as the competition winner
- [SRK](https://www.kaggle.com/sudalairajkumar):  For the starter XGB script which got everyone off on a strong start
- [gdy5](https://www.kaggle.com/guoday):  For showing how much of an impact a single CV feature could have
- [Kazanova](https://www.kaggle.com/kazanova):  For sharing StackNet and the "magic" leak feature

And many others who were generous enough to share their knowledge and establish public baselines for everyone's benefit.

