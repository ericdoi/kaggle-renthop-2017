# kaggle-renthop-2017
A public example Kaggle submission for Two Sigma Connect: Rental Listing Inquiries, 2017

Competition page:  kaggle.com/c/two-sigma-connect-rental-listing-inquiries

This competition was notable for its rich domain-specific features and manageable data size--
aside from a 80GB image dataset. However, the consensus was that image processing yielded little lift over the data in the standard
numeric/text data.

The structure of this repository is modeled after Jeong-Yoon Lee's Kaggler Template (https://github.com/jeongyoonlee/kaggler-template).
It is organized to make model runs reproducible and facilitate ensembling of out-of-fold model predictions.

Makefiles are used to run the modeling pipeline (data prep, model training, prediction, and ensembling).
For overview and motivation, see http://kaggler.com/kagglers-toolbox-setup/.

For modeling, XGB was the main model, ensembled with Logistic Regression.
