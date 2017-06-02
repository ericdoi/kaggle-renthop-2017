# directories
DIR_DATA := ../data
DIR_SRC := ../src

DIR_BUILD := ../build
DIR_FEATURE := $(DIR_BUILD)/feature
DIR_METRIC := $(DIR_BUILD)/metric
DIR_MODEL := $(DIR_BUILD)/model

# Out-of-fold (oof) validation predictions, test predictions
DIR_VAL := $(DIR_BUILD)/val
DIR_TST := $(DIR_BUILD)/tst
DIR_SUB := $(DIR_BUILD)/sub


DATA_TRN := $(DIR_DATA)/train.json
DATA_TST := $(DIR_DATA)/test.json
SAMPLE_SUBMISSION := $(DIR_DATA)/sample_submission.csv

ID_TST := $(DIR_DATA)/id.tst.csv
HEADER := $(DIR_DATA)/header.csv

# labels.  Y_TST contains dummy values
Y_TRN:= $(DIR_FEATURE)/y.trn.txt
Y_TST:= $(DIR_FEATURE)/y.tst.txt


DIRS := $(DIR_DATA) $(DIR_SRC) $(DIR_BUILD) $(DIR_SUB) \
	$(DIR_FEATURE) $(DIR_MODEL) $(DIR_VAL) $(DIR_TST) $(DIR_METRIC)

$(DIRS):
	mkdir -p $@

$(HEADER): $(SAMPLE_SUBMISSION)
	head -1 $< > $@

#$(ID_TST): $(SAMPLE_SUBMISSION)
#	cut -d, -f1 $< | tail -n +2 > $@

#$(Y_TST): $(SAMPLE_SUBMISSION) | $(DIR_FEATURE)
#	cut -d, -f2 $< | tail -n +2 > $@

#$(Y_TRN): $(DATA_TRN) | $(DIR_FEATURE)
#	cut -d, -f132 $< | tail -n +2 > $@

# cleanup
clean::
	find . -name '*.pyc' -delete
