Simple LSTM (10 iterations over dev set):
=========================================

Iteration I
-----------

```
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
```

Iteration II
------------

```
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
```

Iteration III
-------------

```
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
```

Iteration IV (evaluate on train set)
------------------------------------

```
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
```

Iteration V (evaluate on train set after 200 iterations with no dropout)
------------------------------------------------------------------------

```
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
```

Iteration VI (evaluate on train set after 200 iterations, Gal's
----------------------------------------------------------------
dropout (* with error)
----------------------

```
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
```

Iteration VII (evaluate on train set after 200 iterations, na\"ive dropout)
---------------------------------------------------------------------------

```
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
```

Iteration VIII (evaluate on train set after 200 iterations, no dropout)
-----------------------------------------------------------------------

```
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
```

Iteration IX (evaluate on train set after 200 iterations, Gal's dropout)
---------------------------------------------------------------------

```
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
```

Bi-directional LSTM (10 iterations over dev set):
=================================================

Iteration I
-----------

```
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
```

Iteration IV (evaluate on train set)
------------------------------------

```
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
```

Bi-directional LSTM (with separate parameters, 10 iterations over dev set):
===========================================================================

Iteration IV (evaluate on train set)
------------------------------------

```
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
```

TEST RUNS ON DEV
================


Wang XGBoost + Wang LinearSVC
================================================
```
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
```

Wang XGBoost + Wang LinearSVC + Majority
================================================

```
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
```

OPTIONS: --w2v

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6087	recall 0.6087	F1 0.6087
Comparison.Concession                     precision 0.5000	recall 0.1765	F1 0.2609
Comparison.Contrast                       precision 0.9357	recall 0.6452	F1 0.7637
Contingency.Cause.Reason                  precision 0.8333	recall 0.2155	F1 0.3425
Contingency.Cause.Result                  precision 0.2906	recall 0.4722	F1 0.3598
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
EntRel                                    precision 0.3722	recall 0.8605	F1 0.5197
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.8915	recall 0.6197	F1 0.7311
Expansion.Instantiation                   precision 0.3857	recall 0.4737	F1 0.4252
Expansion.Restatement                     precision 0.4091	recall 0.2500	F1 0.3103
Temporal.Asynchronous.Precedence          precision 0.9608	recall 0.6447	F1 0.7717
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6481	F1 0.7778
Temporal.Synchrony                        precision 0.7188	recall 0.8625	F1 0.7841
Overall parser performance --------------
Precision 0.6087 Recall 0.6087 F1 0.6087

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9135	recall 0.9135	F1 0.9135
Comparison.Concession                     precision 0.5000	recall 0.2500	F1 0.3333
Comparison.Contrast                       precision 0.9357	recall 0.9756	F1 0.9552
Contingency.Cause.Reason                  precision 0.8333	recall 0.6579	F1 0.7353
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9672	recall 0.9725	F1 0.9699
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6863	F1 0.8046
Temporal.Synchrony                        precision 0.7188	recall 0.9718	F1 0.8263
Overall parser performance --------------
Precision 0.9135 Recall 0.9135 F1 0.9135

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.3405	recall 0.3405	F1 0.3405
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Result                  precision 0.1782	recall 0.3396	F1 0.2338
EntRel                                    precision 0.3722	recall 0.8605	F1 0.5197
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.4138	recall 0.0976	F1 0.1579
Expansion.Instantiation                   precision 0.2951	recall 0.3750	F1 0.3303
Expansion.Restatement                     precision 0.3607	recall 0.2157	F1 0.2699
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.3405 Recall 0.3405 F1 0.3405
```


OPTIONS: --w2v --lstsq

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5895	recall 0.5895	F1 0.5895
Comparison.Concession                     precision 0.5000	recall 0.1765	F1 0.2609
Comparison.Contrast                       precision 0.9357	recall 0.6452	F1 0.7637
Contingency.Cause.Reason                  precision 0.4123	recall 0.4052	F1 0.4087
Contingency.Cause.Result                  precision 0.5429	recall 0.2639	F1 0.3551
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
EntRel                                    precision 0.3490	recall 0.8279	F1 0.4910
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.8835	recall 0.5948	F1 0.7109
Expansion.Instantiation                   precision 0.2895	recall 0.3860	F1 0.3308
Expansion.Restatement                     precision 0.2453	recall 0.1204	F1 0.1615
Temporal.Asynchronous.Precedence          precision 0.9608	recall 0.6447	F1 0.7717
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6481	F1 0.7778
Temporal.Synchrony                        precision 0.7234	recall 0.8608	F1 0.7861
Overall parser performance --------------
Precision 0.5895 Recall 0.5895 F1 0.5895

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9120	recall 0.9120	F1 0.9120
Comparison.Concession                     precision 0.5000	recall 0.2500	F1 0.3333
Comparison.Contrast                       precision 0.9357	recall 0.9756	F1 0.9552
Contingency.Cause.Reason                  precision 0.8333	recall 0.6579	F1 0.7353
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9572	recall 0.9781	F1 0.9676
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.5000	F1 0.6667
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6863	F1 0.8046
Temporal.Synchrony                        precision 0.7234	recall 0.9714	F1 0.8293
Overall parser performance --------------
Precision 0.9120 Recall 0.9120 F1 0.9120

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.3057	recall 0.3057	F1 0.3057
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 0.2619	recall 0.2821	F1 0.2716
Contingency.Cause.Result                  precision 0.1579	recall 0.0566	F1 0.0833
EntRel                                    precision 0.3490	recall 0.8279	F1 0.4910
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.1579	recall 0.0244	F1 0.0423
Expansion.Instantiation                   precision 0.1940	recall 0.2708	F1 0.2261
Expansion.Restatement                     precision 0.2000	recall 0.0980	F1 0.1316
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.3057 Recall 0.3057 F1 0.3057
```


OPTIONS:
```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5824	recall 0.5824	F1 0.5824
Comparison.Concession                     precision 0.5000	recall 0.1765	F1 0.2609
Comparison.Contrast                       precision 0.9357	recall 0.6452	F1 0.7637
Contingency.Cause.Reason                  precision 0.4730	recall 0.3017	F1 0.3684
Contingency.Cause.Result                  precision 0.7619	recall 0.2222	F1 0.3441
Contingency.Condition                     precision 0.9767	recall 0.9130	F1 0.9438
EntRel                                    precision 0.3463	recall 0.8279	F1 0.4883
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.8945	recall 0.5836	F1 0.7063
Expansion.Instantiation                   precision 0.2474	recall 0.4211	F1 0.3117
Expansion.Restatement                     precision 0.2500	recall 0.1852	F1 0.2128
Temporal.Asynchronous.Precedence          precision 0.9608	recall 0.6447	F1 0.7717
Temporal.Asynchronous.Succession          precision 0.8333	recall 0.6481	F1 0.7292
Temporal.Synchrony                        precision 0.7041	recall 0.8625	F1 0.7753
Overall parser performance --------------
Precision 0.5824 Recall 0.5824 F1 0.5824

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9074	recall 0.9074	F1 0.9074
Comparison.Concession                     precision 0.5000	recall 0.2500	F1 0.3333
Comparison.Contrast                       precision 0.9357	recall 0.9756	F1 0.9552
Contingency.Cause.Reason                  precision 0.8621	recall 0.6579	F1 0.7463
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9767	recall 0.9130	F1 0.9438
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9516	recall 0.9725	F1 0.9620
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.3333	F1 0.5000
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6863	F1 0.8046
Temporal.Synchrony                        precision 0.7041	recall 0.9718	F1 0.8166
Overall parser performance --------------
Precision 0.9074 Recall 0.9074 F1 0.9074

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.2964	recall 0.2964	F1 0.2964
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 0.2222	recall 0.1282	F1 0.1626
Contingency.Cause.Result                  precision 0.0000	recall 0.0000	F1 0.0000
EntRel                                    precision 0.3463	recall 0.8279	F1 0.4883
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.0769	recall 0.0081	F1 0.0147
Expansion.Instantiation                   precision 0.1705	recall 0.3125	F1 0.2206
Expansion.Restatement                     precision 0.2308	recall 0.1765	F1 0.2000
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Asynchronous.Succession          precision 0.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.2964 Recall 0.2964 F1 0.2964
```


OPTIONS: --w2v
```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6037	recall 0.6037	F1 0.6037
Comparison.Concession                     precision 0.4000	recall 0.1176	F1 0.1818
Comparison.Contrast                       precision 0.9405	recall 0.6371	F1 0.7596
Contingency.Cause.Reason                  precision 0.6833	recall 0.3504	F1 0.4633
Contingency.Cause.Result                  precision 0.2484	recall 0.5278	F1 0.3378
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
EntRel                                    precision 0.3870	recall 0.8047	F1 0.5227
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.9421	recall 0.5869	F1 0.7232
Expansion.Instantiation                   precision 0.3810	recall 0.5614	F1 0.4539
Expansion.Restatement                     precision 0.3750	recall 0.1667	F1 0.2308
Temporal.Asynchronous.Precedence          precision 0.9608	recall 0.6447	F1 0.7717
Temporal.Asynchronous.Succession          precision 0.7600	recall 0.7037	F1 0.7308
Temporal.Synchrony                        precision 0.7083	recall 0.8608	F1 0.7771
Overall parser performance --------------
Precision 0.6037 Recall 0.6037 F1 0.6037

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9150	recall 0.9150	F1 0.9150
Comparison.Concession                     precision 0.4000	recall 0.1667	F1 0.2353
Comparison.Contrast                       precision 0.9405	recall 0.9634	F1 0.9518
Contingency.Cause.Reason                  precision 0.9310	recall 0.6923	F1 0.7941
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9672	recall 0.9725	F1 0.9699
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 0.8333	recall 0.8333	F1 0.8333
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9500	recall 0.7451	F1 0.8352
Temporal.Synchrony                        precision 0.7083	recall 0.9714	F1 0.8193
Overall parser performance --------------
Precision 0.9150 Recall 0.9150 F1 0.9150

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.3298	recall 0.3298	F1 0.3298
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 0.4516	recall 0.1795	F1 0.2569
Contingency.Cause.Result                  precision 0.1606	recall 0.4151	F1 0.2316
EntRel                                    precision 0.3870	recall 0.8047	F1 0.5227
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.2857	recall 0.0163	F1 0.0308
Expansion.Instantiation                   precision 0.3067	recall 0.4792	F1 0.3740
Expansion.Restatement                     precision 0.3095	recall 0.1275	F1 0.1806
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Asynchronous.Succession          precision 0.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.3298 Recall 0.3298 F1 0.3298
```


OPTIONS: --w2v --lstsq
```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6044	recall 0.6044	F1 0.6044
Comparison.Concession                     precision 0.5000	recall 0.1765	F1 0.2609
Comparison.Contrast                       precision 0.9357	recall 0.6452	F1 0.7637
Contingency.Cause.Reason                  precision 0.6167	recall 0.3190	F1 0.4205
Contingency.Cause.Result                  precision 0.2759	recall 0.4384	F1 0.3386
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
EntRel                                    precision 0.4077	recall 0.7907	F1 0.5380
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.9545	recall 0.6217	F1 0.7530
Expansion.Instantiation                   precision 0.3061	recall 0.5263	F1 0.3871
Expansion.Restatement                     precision 0.2561	recall 0.1944	F1 0.2211
Temporal.Asynchronous.Precedence          precision 0.9608	recall 0.6533	F1 0.7778
Temporal.Asynchronous.Succession          precision 0.6364	recall 0.6481	F1 0.6422
Temporal.Synchrony                        precision 0.7245	recall 0.8765	F1 0.7933
Overall parser performance --------------
Precision 0.6044 Recall 0.6044 F1 0.6044

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9105	recall 0.9105	F1 0.9105
Comparison.Concession                     precision 0.5000	recall 0.2500	F1 0.3333
Comparison.Contrast                       precision 0.9357	recall 0.9756	F1 0.9552
Contingency.Cause.Reason                  precision 0.8333	recall 0.6579	F1 0.7353
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9615	recall 0.9669	F1 0.9642
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 0.7500	recall 0.5000	F1 0.6000
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6863	F1 0.8046
Temporal.Synchrony                        precision 0.7245	recall 0.9861	F1 0.8353
Overall parser performance --------------
Precision 0.9105 Recall 0.9105 F1 0.9105

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.3351	recall 0.3351	F1 0.3351
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 0.4000	recall 0.1538	F1 0.2222
Contingency.Cause.Result                  precision 0.1600	recall 0.2963	F1 0.2078
EntRel                                    precision 0.4077	recall 0.7907	F1 0.5380
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.8750	recall 0.1138	F1 0.2014
Expansion.Instantiation                   precision 0.2360	recall 0.4375	F1 0.3066
Expansion.Restatement                     precision 0.2308	recall 0.1765	F1 0.2000
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Asynchronous.Succession          precision 0.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.3351 Recall 0.3351 F1 0.3351
```


OPTIONS:
```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6193	recall 0.6193	F1 0.6193
Comparison.Concession                     precision 0.4286	recall 0.1765	F1 0.2500
Comparison.Contrast                       precision 0.9353	recall 0.6411	F1 0.7608
Contingency.Cause.Reason                  precision 0.4264	recall 0.4741	F1 0.4490
Contingency.Cause.Result                  precision 0.6364	recall 0.2917	F1 0.4000
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
EntRel                                    precision 0.3903	recall 0.9023	F1 0.5449
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.9552	recall 0.6316	F1 0.7604
Expansion.Instantiation                   precision 0.3125	recall 0.5263	F1 0.3922
Expansion.Restatement                     precision 0.2581	recall 0.0741	F1 0.1151
Temporal.Asynchronous.Precedence          precision 0.9259	recall 0.6579	F1 0.7692
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6481	F1 0.7778
Temporal.Synchrony                        precision 0.7245	recall 0.8765	F1 0.7933
Overall parser performance --------------
Precision 0.6193 Recall 0.6193 F1 0.6193

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9135	recall 0.9135	F1 0.9135
Comparison.Concession                     precision 0.4286	recall 0.2500	F1 0.3158
Comparison.Contrast                       precision 0.9353	recall 0.9695	F1 0.9521
Contingency.Cause.Reason                  precision 0.8333	recall 0.6579	F1 0.7353
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9724	recall 0.9724	F1 0.9724
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6863	F1 0.8046
Temporal.Synchrony                        precision 0.7245	recall 0.9861	F1 0.8353
Overall parser performance --------------
Precision 0.9135 Recall 0.9135 F1 0.9135

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.3605	recall 0.3605	F1 0.3605
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 0.3030	recall 0.3846	F1 0.3390
Contingency.Cause.Result                  precision 0.2941	recall 0.0943	F1 0.1429
EntRel                                    precision 0.3903	recall 0.9023	F1 0.5449
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.8000	recall 0.1301	F1 0.2238
Expansion.Instantiation                   precision 0.2414	recall 0.4375	F1 0.3111
Expansion.Restatement                     precision 0.1154	recall 0.0294	F1 0.0469
Temporal.Asynchronous.Precedence          precision 0.3333	recall 0.0370	F1 0.0667
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.3605 Recall 0.3605 F1 0.3605
```


OPTIONS: --w2v
```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5966	recall 0.5966	F1 0.5966
Comparison.Concession                     precision 0.5000	recall 0.1765	F1 0.2609
Comparison.Contrast                       precision 0.9195	recall 0.6452	F1 0.7583
Contingency.Cause.Reason                  precision 0.9394	recall 0.2650	F1 0.4133
Contingency.Cause.Result                  precision 0.1928	recall 0.6575	F1 0.2981
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
EntRel                                    precision 0.3867	recall 0.8093	F1 0.5233
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.9087	recall 0.6176	F1 0.7354
Expansion.Instantiation                   precision 1.0000	recall 0.1579	F1 0.2727
Expansion.Restatement                     precision 0.3953	recall 0.1574	F1 0.2252
Temporal.Asynchronous.Precedence          precision 0.9615	recall 0.6667	F1 0.7874
Temporal.Asynchronous.Succession          precision 0.9744	recall 0.7037	F1 0.8172
Temporal.Synchrony                        precision 0.7528	recall 0.8590	F1 0.8024
Overall parser performance --------------
Precision 0.5966 Recall 0.5966 F1 0.5966

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9226	recall 0.9226	F1 0.9226
Comparison.Concession                     precision 0.5000	recall 0.2500	F1 0.3333
Comparison.Contrast                       precision 0.9357	recall 0.9756	F1 0.9552
Contingency.Cause.Reason                  precision 0.9375	recall 0.7692	F1 0.8451
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9674	recall 0.9727	F1 0.9700
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 0.6667	recall 0.6667	F1 0.6667
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9744	recall 0.7451	F1 0.8444
Temporal.Synchrony                        precision 0.7528	recall 0.9710	F1 0.8481
Overall parser performance --------------
Precision 0.9226 Recall 0.9226 F1 0.9226

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.3097	recall 0.3097	F1 0.3097
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 1.0000	recall 0.0128	F1 0.0253
Contingency.Cause.Result                  precision 0.1373	recall 0.5926	F1 0.2230
EntRel                                    precision 0.3867	recall 0.8093	F1 0.5233
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.4583	recall 0.0894	F1 0.1497
Expansion.Instantiation                   precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Restatement                     precision 0.3514	recall 0.1275	F1 0.1871
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.0385	F1 0.0741
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.3097 Recall 0.3097 F1 0.3097
```

OPTIONS: --w2v --lstsq
```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6001	recall 0.6001	F1 0.6001
Comparison.Concession                     precision 0.5000	recall 0.1765	F1 0.2609
Comparison.Contrast                       precision 0.9357	recall 0.6452	F1 0.7637
Contingency.Cause.Reason                  precision 0.6744	recall 0.2500	F1 0.3648
Contingency.Cause.Result                  precision 0.2312	recall 0.5479	F1 0.3252
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
EntRel                                    precision 0.4448	recall 0.6744	F1 0.5360
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.9471	recall 0.6459	F1 0.7680
Expansion.Instantiation                   precision 0.2947	recall 0.4912	F1 0.3684
Expansion.Restatement                     precision 0.2595	recall 0.3148	F1 0.2845
Temporal.Asynchronous.Precedence          precision 0.7612	recall 0.6800	F1 0.7183
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6481	F1 0.7778
Temporal.Synchrony                        precision 0.7188	recall 0.8625	F1 0.7841
Overall parser performance --------------
Precision 0.6001 Recall 0.6001 F1 0.6001

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9135	recall 0.9135	F1 0.9135
Comparison.Concession                     precision 0.5000	recall 0.2500	F1 0.3333
Comparison.Contrast                       precision 0.9357	recall 0.9756	F1 0.9552
Contingency.Cause.Reason                  precision 0.8333	recall 0.6579	F1 0.7353
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9672	recall 0.9725	F1 0.9699
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6863	F1 0.8046
Temporal.Synchrony                        precision 0.7188	recall 0.9718	F1 0.8263
Overall parser performance --------------
Precision 0.9135 Recall 0.9135 F1 0.9135

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.3244	recall 0.3244	F1 0.3244
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 0.3077	recall 0.0513	F1 0.0879
Contingency.Cause.Result                  precision 0.1529	recall 0.4444	F1 0.2275
EntRel                                    precision 0.4448	recall 0.6744	F1 0.5360
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.8000	recall 0.1626	F1 0.2703
Expansion.Instantiation                   precision 0.2209	recall 0.3958	F1 0.2836
Expansion.Restatement                     precision 0.2302	recall 0.2843	F1 0.2544
Temporal.Asynchronous.Precedence          precision 0.1250	recall 0.0769	F1 0.0952
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.3244 Recall 0.3244 F1 0.3244
```


OPTIONS:
```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6108	recall 0.6108	F1 0.6108
Comparison.Concession                     precision 0.5000	recall 0.1765	F1 0.2609
Comparison.Contrast                       precision 0.9357	recall 0.6452	F1 0.7637
Contingency.Cause.Reason                  precision 0.5063	recall 0.3448	F1 0.4103
Contingency.Cause.Result                  precision 0.2778	recall 0.5556	F1 0.3704
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
EntRel                                    precision 0.4104	recall 0.8093	F1 0.5446
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.9014	recall 0.6295	F1 0.7413
Expansion.Instantiation                   precision 0.3131	recall 0.5439	F1 0.3974
Expansion.Restatement                     precision 0.3939	recall 0.1204	F1 0.1844
Temporal.Asynchronous.Precedence          precision 0.9608	recall 0.6447	F1 0.7717
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6481	F1 0.7778
Temporal.Synchrony                        precision 0.7188	recall 0.8625	F1 0.7841
Overall parser performance --------------
Precision 0.6108 Recall 0.6108 F1 0.6108

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9135	recall 0.9135	F1 0.9135
Comparison.Concession                     precision 0.5000	recall 0.2500	F1 0.3333
Comparison.Contrast                       precision 0.9357	recall 0.9756	F1 0.9552
Contingency.Cause.Reason                  precision 0.8333	recall 0.6579	F1 0.7353
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9773	recall 0.9348	F1 0.9556
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9672	recall 0.9725	F1 0.9699
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 0.9722	recall 0.6863	F1 0.8046
Temporal.Synchrony                        precision 0.7188	recall 0.9718	F1 0.8263
Overall parser performance --------------
Precision 0.9135 Recall 0.9135 F1 0.9135

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.3445	recall 0.3445	F1 0.3445
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 1.0000	recall 0.0000	F1 0.0000
Contingency.Cause.Reason                  precision 0.3061	recall 0.1923	F1 0.2362
Contingency.Cause.Result                  precision 0.1875	recall 0.4528	F1 0.2652
EntRel                                    precision 0.4104	recall 0.8093	F1 0.5446
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.5000	recall 0.1220	F1 0.1961
Expansion.Instantiation                   precision 0.2444	recall 0.4583	F1 0.3188
Expansion.Restatement                     precision 0.2857	recall 0.0784	F1 0.1231
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.0000	F1 0.0000
Overall parser performance --------------
Precision 0.3445 Recall 0.3445 F1 0.3445
```
