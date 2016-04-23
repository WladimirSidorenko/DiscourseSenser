Simple LSTM (10 iterations over dev set):
=========================================

Iteration I
-----------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.7443 Recall 0.7443 F1 0.7443

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.9211 Recall 0.9211 F1 0.9211

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.5888 Recall 0.5888 F1 0.5888

Iteration II
------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.7330 Recall 0.7330 F1 0.7330

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.9014 Recall 0.9014 F1 0.9014

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.5848 Recall 0.5848 F1 0.5848

Iteration III
-------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.7188 Recall 0.7188 F1 0.7188

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8953 Recall 0.8953 F1 0.8953

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.5634 Recall 0.5634 F1 0.5634

Iteration IV (evaluate on train set)
------------------------------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.5010 Recall 0.5010 F1 0.5010

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.7943 Recall 0.7804 F1 0.7873

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.2582 Recall 0.2620 F1 0.2601

Iteration V (evaluate on train set after 200 iterations with no dropout)
------------------------------------------------------------------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.4668 Recall 0.4584 F1 0.4626

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.7600 Recall 0.7224 F1 0.7408

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.2311 Recall 0.2330 F1 0.2321


Iteration VI (evaluate on train set after 200 iterations, Gal's
----------------------------------------------------------------
dropout (* with error)
----------------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.5069 Recall 0.5069 F1 0.5069

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8165 Recall 0.8021 F1 0.8092

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.2513 Recall 0.2550 F1 0.2531

Iteration VII (evaluate on train set after 200 iterations, na\"ive dropout)
---------------------------------------------------------------------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.4999 Recall 0.4999 F1 0.4999

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8029 Recall 0.7888 F1 0.7958

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.2486 Recall 0.2522 F1 0.2504

Iteration VIII (evaluate on train set after 200 iterations, no dropout)
-----------------------------------------------------------------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.4784 Recall 0.4744 F1 0.4764

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.7752 Recall 0.7474 F1 0.7611

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.2380 Recall 0.2414 F1 0.2397

Iteration IX (evaluate on train set after 200 iterations, Gal's dropout)
---------------------------------------------------------------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.4908 Recall 0.4908 F1 0.4908

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.7779 Recall 0.7642 F1 0.7710

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.2525 Recall 0.2561 F1 0.2543


Bi-directional LSTM (10 iterations over dev set):
=================================================

Iteration I
-----------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6307 Recall 0.6307 F1 0.6307

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8938 Recall 0.8938 F1 0.8938

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.3992 Recall 0.3992 F1 0.3992

Iteration IV (evaluate on train set)
------------------------------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.4961 Recall 0.4961 F1 0.4961

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8079 Recall 0.7937 F1 0.8007

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.2382 Recall 0.2416 F1 0.2399

Bi-directional LSTM (with separate parameters, 10 iterations over dev set):
===========================================================================

Iteration IV (evaluate on train set)
------------------------------------

================================================
Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.4919 Recall 0.4919 F1 0.4919

================================================
Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8007 Recall 0.7866 F1 0.7936

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.2366 Recall 0.2400 F1 0.2382

TEST RUNS ON DEV
================


Wang XGBoost + Wang LinearSVC
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6574	recall 0.6570	F1 0.6572
Comparison.Concession                     precision 0.4545	recall 0.2941	F1 0.3571
Comparison.Contrast                       precision 0.9043	recall 0.6800	F1 0.7763
Contingency.Cause.Reason                  precision 0.5273	recall 0.4833	F1 0.5043
Contingency.Cause.Result                  precision 0.5833	recall 0.2917	F1 0.3889
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
EntRel                                    precision 0.4631	recall 0.8744	F1 0.6055
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.7590	recall 0.7565	F1 0.7577
Expansion.Exception                       precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Instantiation                   precision 0.8125	recall 0.2281	F1 0.3562
Expansion.Restatement                     precision 0.3611	recall 0.3578	F1 0.3594
Temporal.Asynchronous.Precedence          precision 0.9412	recall 0.6316	F1 0.7559
Temporal.Asynchronous.Succession          precision 0.9286	recall 0.7222	F1 0.8125
Temporal.Synchrony                        precision 0.7467	recall 0.8116	F1 0.7778
Overall parser performance --------------
Precision 0.6574 Recall 0.6570 F1 0.6572

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9105	recall 0.9105	F1 0.9105
Comparison.Concession                     precision 0.4545	recall 0.4167	F1 0.4348
Comparison.Contrast                       precision 0.9458	recall 0.9573	F1 0.9515
Contingency.Cause.Reason                  precision 0.7955	recall 0.8140	F1 0.8046
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
EntRel                                    precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9624	recall 0.9728	F1 0.9676
Expansion.Exception                       precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 0.7143	recall 0.8333	F1 0.7692
Temporal.Asynchronous.Precedence          precision 0.9583	recall 0.9388	F1 0.9485
Temporal.Asynchronous.Succession          precision 0.9268	recall 0.7451	F1 0.8261
Temporal.Synchrony                        precision 0.7432	recall 0.8730	F1 0.8029
Overall parser performance --------------
Precision 0.9105 Recall 0.9105 F1 0.9105

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.4345	recall 0.4339	F1 0.4342
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.5909	recall 0.1512	F1 0.2407
Contingency.Cause.Reason                  precision 0.3485	recall 0.2987	F1 0.3217
Contingency.Cause.Result                  precision 0.2500	recall 0.0943	F1 0.1370
Contingency.Condition                     precision 1.0000	recall 1.0000	F1 1.0000
EntRel                                    precision 0.4631	recall 0.8744	F1 0.6055
Expansion.Alternative                     precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.4463	recall 0.4355	F1 0.4408
Expansion.Exception                       precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Instantiation                   precision 0.5714	recall 0.0833	F1 0.1455
Expansion.Restatement                     precision 0.3366	recall 0.3301	F1 0.3333
Temporal.Asynchronous.Precedence          precision 0.6667	recall 0.0741	F1 0.1333
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.3333	F1 0.5000
Temporal.Synchrony                        precision 1.0000	recall 0.1667	F1 0.2857
Overall parser performance --------------
Precision 0.4345 Recall 0.4339 F1 0.4342


Wang XGBoost + Wang LinearSVC + Majority
================================================

================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6574	recall 0.6570	F1 0.6572
Comparison.Concession                     precision 0.4545	recall 0.2941	F1 0.3571
Comparison.Contrast                       precision 0.9043	recall 0.6800	F1 0.7763
Contingency.Cause.Reason                  precision 0.5421	recall 0.4833	F1 0.5110
Contingency.Cause.Result                  precision 0.7407	recall 0.2778	F1 0.4040
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
EntRel                                    precision 0.4487	recall 0.8744	F1 0.5931
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.7607	recall 0.7532	F1 0.7569
Expansion.Exception                       precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Instantiation                   precision 0.8125	recall 0.2281	F1 0.3562
Expansion.Restatement                     precision 0.3611	recall 0.3578	F1 0.3594
Temporal.Asynchronous.Precedence          precision 0.9434	recall 0.6579	F1 0.7752
Temporal.Asynchronous.Succession          precision 0.9512	recall 0.7222	F1 0.8211
Temporal.Synchrony                        precision 0.7467	recall 0.8116	F1 0.7778
Overall parser performance --------------
Precision 0.6574 Recall 0.6570 F1 0.6572

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9135	recall 0.9135	F1 0.9135
Comparison.Concession                     precision 0.4545	recall 0.4167	F1 0.4348
Comparison.Contrast                       precision 0.9458	recall 0.9573	F1 0.9515
Contingency.Cause.Reason                  precision 0.8140	recall 0.8140	F1 0.8140
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
EntRel                                    precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9624	recall 0.9728	F1 0.9676
Expansion.Exception                       precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 0.7143	recall 0.8333	F1 0.7692
Temporal.Asynchronous.Precedence          precision 0.9600	recall 0.9796	F1 0.9697
Temporal.Asynchronous.Succession          precision 0.9500	recall 0.7451	F1 0.8352
Temporal.Synchrony                        precision 0.7432	recall 0.8730	F1 0.8029
Overall parser performance --------------
Precision 0.9135 Recall 0.9135 F1 0.9135

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.4318	recall 0.4312	F1 0.4315
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.5909	recall 0.1512	F1 0.2407
Contingency.Cause.Reason                  precision 0.3594	recall 0.2987	F1 0.3262
Contingency.Cause.Result                  precision 0.3636	recall 0.0755	F1 0.1250
Contingency.Condition                     precision 1.0000	recall 1.0000	F1 1.0000
EntRel                                    precision 0.4487	recall 0.8744	F1 0.5931
Expansion.Alternative                     precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.4454	recall 0.4274	F1 0.4362
Expansion.Exception                       precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Instantiation                   precision 0.5714	recall 0.0833	F1 0.1455
Expansion.Restatement                     precision 0.3366	recall 0.3301	F1 0.3333
Temporal.Asynchronous.Precedence          precision 0.6667	recall 0.0741	F1 0.1333
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.3333	F1 0.5000
Temporal.Synchrony                        precision 1.0000	recall 0.1667	F1 0.2857
Overall parser performance --------------
Precision 0.4318 Recall 0.4312 F1 0.4315
