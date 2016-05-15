MD5 SUMS ORIG
=============
```
fd1f57f032b92c904f47280cb07bf8e6  Projects/DiscourseSenserBkp.2016.05.07/dsenser/data/models/pdtb.sense.model
8c564ed6b99b20e4b9313bca0b3fa6be  Projects/DiscourseSenserBkp.2016.05.07/dsenser/data/models/pdtb.sense.model.LSTMSenser
7ded99f5d2f96f2ae4c0ae401110874f  Projects/DiscourseSenserBkp.2016.05.07/dsenser/data/models/pdtb.sense.model.MajorSenser
c852915080e526141beeec0fd02d9085  Projects/DiscourseSenserBkp.2016.05.07/dsenser/data/models/pdtb.sense.model.SVDSenser
7ecb67c8c3fea11a93f2a15aa7e0e29f  Projects/DiscourseSenserBkp.2016.05.07/dsenser/data/models/pdtb.sense.model.WangSenser
da54f403c0ddf6ecdc890b7889e0c142  Projects/DiscourseSenserBkp.2016.05.07/dsenser/data/models/pdtb.sense.model.XGBoostSenser
```

MD5 SUMS RECAST
===============
```
99f8ed08c0494658e211f3467fdbd19d  dsenser/data/models/pdtb.sense.model
3f777a9786673ba01387ba65aa0b013c  dsenser/data/models/pdtb.sense.model.LSTMSenser
7ded99f5d2f96f2ae4c0ae401110874f  dsenser/data/models/pdtb.sense.model.MajorSenser
c852915080e526141beeec0fd02d9085  dsenser/data/models/pdtb.sense.model.SVDSenser
7ecb67c8c3fea11a93f2a15aa7e0e29f  dsenser/data/models/pdtb.sense.model.WangSenser
da54f403c0ddf6ecdc890b7889e0c142  dsenser/data/models/pdtb.sense.model.XGBoostSenser
```

Recast Procedure:
=================
```python
        from theano.sandbox.cuda.var import CudaNdarraySharedVariable
        from theano.compile.sharedvalue import SharedVariable


        def recast_cuda_vars(a_inst):
            for attr in dir(a_inst):
                if not isinstance(getattr(a_inst, attr),
                                  CudaNdarraySharedVariable):
                    continue
                iattr = getattr(a_inst, attr)
                setattr(a_inst, attr, SharedVariable(name=str(iattr),
                                                     type=iattr.type,
                                                     value=iattr.get_value(),
                                                     strict=False,
                                                     allow_downcast=None))

        for ipath in self.model_paths:
            print("ipath =", repr(ipath), file=sys.stderr)
            if not ipath.endswith("LSTMSenser") and \
               not ipath.endswith("SVDSenser"):
                continue
            with open(ipath, "rb") as ifile:
                imodel = load(ifile)
            recast_cuda_vars(imodel.explicit)
            recast_cuda_vars(imodel.implicit)
            with open(ipath, "wb") as ofile:
                dump(imodel, ofile)
            continue
```

DEV SET (all models)
====================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.7188	recall 0.7188	F1 0.7188
Comparison.Concession                     precision 0.5455	recall 0.3529	F1 0.4286
Comparison.Contrast                       precision 0.9309	recall 0.7000	F1 0.7991
Contingency.Cause.Reason                  precision 0.7568	recall 0.6512	F1 0.7000
Contingency.Cause.Result                  precision 0.7000	recall 0.3889	F1 0.5000
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
EntRel                                    precision 0.4780	recall 0.9581	F1 0.6378
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.8221	recall 0.7929	F1 0.8072
Expansion.Instantiation                   precision 1.0000	recall 0.3158	F1 0.4800
Expansion.Restatement                     precision 0.5059	recall 0.3945	F1 0.4433
Temporal.Asynchronous.Precedence          precision 0.9649	recall 0.7237	F1 0.8271
Temporal.Asynchronous.Succession          precision 0.9512	recall 0.7647	F1 0.8478
Temporal.Synchrony                        precision 0.8169	recall 0.9355	F1 0.8722
Overall parser performance --------------
Precision 0.7188 Recall 0.7188 F1 0.7188

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9439	recall 0.9439	F1 0.9439
Comparison.Concession                     precision 0.5455	recall 0.5000	F1 0.5217
Comparison.Contrast                       precision 0.9518	recall 0.9634	F1 0.9576
Contingency.Cause.Reason                  precision 0.9796	recall 0.9231	F1 0.9505
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9679	recall 0.9837	F1 0.9757
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.8125	F1 0.8966
Temporal.Synchrony                        precision 0.8116	recall 0.9825	F1 0.8889
Overall parser performance --------------
Precision 0.9439 Recall 0.9439 F1 0.9439

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5207	recall 0.5207	F1 0.5207
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.7727	recall 0.1977	F1 0.3148
Contingency.Cause.Reason                  precision 0.5806	recall 0.4675	F1 0.5180
Contingency.Cause.Result                  precision 0.5000	recall 0.2264	F1 0.3117
EntRel                                    precision 0.4780	recall 0.9581	F1 0.6378
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.5766	recall 0.5120	F1 0.5424
Expansion.Instantiation                   precision 1.0000	recall 0.1875	F1 0.3158
Expansion.Restatement                     precision 0.4750	recall 0.3689	F1 0.4153
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.2222	F1 0.3636
Temporal.Asynchronous.Succession          precision 0.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.4000	F1 0.5714
Overall parser performance --------------
Precision 0.5207 Recall 0.5207 F1 0.5207
```

TRAIN SET (all models)
======================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6435	recall 0.6435	F1 0.6435
Comparison.Concession                     precision 0.6803	recall 0.2604	F1 0.3766
Comparison.Contrast                       precision 0.7627	recall 0.6551	F1 0.7048
Contingency.Cause.Reason                  precision 0.6668	recall 0.5781	F1 0.6193
Contingency.Cause.Result                  precision 0.8333	recall 0.3769	F1 0.5190
Contingency.Condition                     precision 0.9581	recall 0.8841	F1 0.9196
EntRel                                    precision 0.3790	recall 0.9506	F1 0.5420
Expansion.Alternative                     precision 0.9482	recall 0.9104	F1 0.9289
Expansion.Alternative.Chosen alternative  precision 0.9606	recall 0.5083	F1 0.6649
Expansion.Conjunction                     precision 0.7911	recall 0.7152	F1 0.7513
Expansion.Exception                       precision 0.9000	recall 0.6000	F1 0.7200
Expansion.Instantiation                   precision 0.9173	recall 0.4206	F1 0.5768
Expansion.Restatement                     precision 0.5238	recall 0.3218	F1 0.3987
Temporal.Asynchronous.Precedence          precision 0.9255	recall 0.6699	F1 0.7773
Temporal.Asynchronous.Succession          precision 0.9735	recall 0.6742	F1 0.7966
Temporal.Synchrony                        precision 0.7214	recall 0.7781	F1 0.7487
Overall parser performance --------------
Precision 0.6435 Recall 0.6435 F1 0.6435

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 2 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg1 Arg2 extractor combined : Precision 0.9986 Recall 1.0000 F1 0.9993
Sense classification--------------
*Micro-Average                            precision 0.8764	recall 0.8776	F1 0.8770
Comparison.Concession                     precision 0.6791	recall 0.3043	F1 0.4202
Comparison.Contrast                       precision 0.7587	recall 0.9271	F1 0.8345
Contingency.Cause.Reason                  precision 0.9583	recall 0.9312	F1 0.9446
Contingency.Cause.Result                  precision 0.9931	recall 0.9128	F1 0.9512
Contingency.Condition                     precision 0.9580	recall 0.8846	F1 0.9198
Expansion.Alternative                     precision 0.9462	recall 0.9263	F1 0.9362
Expansion.Alternative.Chosen alternative  precision 0.9600	recall 0.9796	F1 0.9697
Expansion.Conjunction                     precision 0.9590	recall 0.9680	F1 0.9635
Expansion.Exception                       precision 0.8889	recall 0.6154	F1 0.7273
Expansion.Instantiation                   precision 0.9872	recall 0.9788	F1 0.9830
Expansion.Restatement                     precision 0.8247	recall 0.6667	F1 0.7373
Temporal.Asynchronous.Precedence          precision 0.9215	recall 0.9804	F1 0.9500
Temporal.Asynchronous.Succession          precision 0.9758	recall 0.7727	F1 0.8625
Temporal.Synchrony                        precision 0.7188	recall 0.8700	F1 0.7872
Overall parser performance --------------
Precision 0.8764 Recall 0.8776 F1 0.8770

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg 2 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 0.9988 F1 0.9994
Sense classification--------------
*Micro-Average                            precision 0.4523	recall 0.4518	F1 0.4521
Comparison.Concession                     precision 0.8000	recall 0.0203	F1 0.0396
Comparison.Contrast                       precision 0.8043	recall 0.1599	F1 0.2667
Contingency.Cause.Reason                  precision 0.5069	recall 0.4149	F1 0.4563
Contingency.Cause.Result                  precision 0.6859	recall 0.2112	F1 0.3229
Contingency.Condition                     precision 1.0000	recall 0.7500	F1 0.8571
EntRel                                    precision 0.3790	recall 0.9506	F1 0.5420
Expansion.Alternative                     precision 1.0000	recall 0.6364	F1 0.7778
Expansion.Alternative.Chosen alternative  precision 0.9630	recall 0.1831	F1 0.3077
Expansion.Conjunction                     precision 0.4981	recall 0.3818	F1 0.4322
Expansion.Exception                       precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Instantiation                   precision 0.8771	recall 0.3072	F1 0.4551
Expansion.Restatement                     precision 0.5049	recall 0.3056	F1 0.3807
Temporal.Asynchronous.Precedence          precision 0.9437	recall 0.1463	F1 0.2533
Temporal.Asynchronous.Succession          precision 0.8125	recall 0.0909	F1 0.1635
Temporal.Synchrony                        precision 1.0000	recall 0.0863	F1 0.1589
Overall parser performance --------------
Precision 0.4523 Recall 0.4518 F1 0.4521
```

DEV SET (no LSTM)
=================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.7067	recall 0.7067	F1 0.7067
Comparison.Concession                     precision 0.5455	recall 0.3529	F1 0.4286
Comparison.Contrast                       precision 0.9309	recall 0.7000	F1 0.7991
Contingency.Cause.Reason                  precision 0.6720	recall 0.6412	F1 0.6563
Contingency.Cause.Result                  precision 0.6486	recall 0.3333	F1 0.4404
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
EntRel                                    precision 0.4950	recall 0.9209	F1 0.6439
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.7830	recall 0.8058	F1 0.7943
Expansion.Instantiation                   precision 0.9375	recall 0.2632	F1 0.4110
Expansion.Restatement                     precision 0.4752	recall 0.4404	F1 0.4571
Temporal.Asynchronous.Precedence          precision 0.9636	recall 0.6974	F1 0.8092
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.7647	F1 0.8667
Temporal.Synchrony                        precision 0.8000	recall 0.8000	F1 0.8000
Overall parser performance --------------
Precision 0.7067 Recall 0.7067 F1 0.7067

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9302	recall 0.9302	F1 0.9302
Comparison.Concession                     precision 0.5455	recall 0.5000	F1 0.5217
Comparison.Contrast                       precision 0.9518	recall 0.9634	F1 0.9576
Contingency.Cause.Reason                  precision 0.8333	recall 0.9259	F1 0.8772
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9628	recall 0.9837	F1 0.9731
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9600	recall 0.9796	F1 0.9697
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.7917	F1 0.8837
Temporal.Synchrony                        precision 0.7931	recall 0.8364	F1 0.8142
Overall parser performance --------------
Precision 0.9302 Recall 0.9302 F1 0.9302

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5100	recall 0.5100	F1 0.5100
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.7727	recall 0.1977	F1 0.3148
Contingency.Cause.Reason                  precision 0.5231	recall 0.4416	F1 0.4789
Contingency.Cause.Result                  precision 0.3810	recall 0.1509	F1 0.2162
EntRel                                    precision 0.4950	recall 0.9209	F1 0.6439
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.5231	recall 0.5440	F1 0.5333
Expansion.Instantiation                   precision 0.8571	recall 0.1250	F1 0.2182
Expansion.Restatement                     precision 0.4479	recall 0.4175	F1 0.4322
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.1852	F1 0.3125
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.3333	F1 0.5000
Temporal.Synchrony                        precision 1.0000	recall 0.4000	F1 0.5714
Overall parser performance --------------
Precision 0.5100 Recall 0.5100 F1 0.5100
```

TRAIN SET (no LSTM)
===================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6382	recall 0.6382	F1 0.6382
Comparison.Concession                     precision 0.6723	recall 0.2471	F1 0.3613
Comparison.Contrast                       precision 0.7547	recall 0.6467	F1 0.6965
Contingency.Cause.Reason                  precision 0.6169	recall 0.5739	F1 0.5946
Contingency.Cause.Result                  precision 0.8085	recall 0.3734	F1 0.5108
Contingency.Condition                     precision 0.9581	recall 0.8833	F1 0.9192
EntRel                                    precision 0.3987	recall 0.8938	F1 0.5515
Expansion.Alternative                     precision 0.9534	recall 0.9109	F1 0.9316
Expansion.Alternative.Chosen alternative  precision 0.9609	recall 0.5146	F1 0.6703
Expansion.Conjunction                     precision 0.7477	recall 0.7393	F1 0.7434
Expansion.Exception                       precision 0.9000	recall 0.6000	F1 0.7200
Expansion.Instantiation                   precision 0.8981	recall 0.3784	F1 0.5325
Expansion.Restatement                     precision 0.4626	recall 0.3477	F1 0.3970
Temporal.Asynchronous.Precedence          precision 0.9251	recall 0.6765	F1 0.7815
Temporal.Asynchronous.Succession          precision 0.9682	recall 0.6844	F1 0.8019
Temporal.Synchrony                        precision 0.7072	recall 0.7181	F1 0.7126
Overall parser performance --------------
Precision 0.6382 Recall 0.6382 F1 0.6382

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 2 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg1 Arg2 extractor combined : Precision 0.9986 Recall 1.0000 F1 0.9993
Sense classification--------------
*Micro-Average                            precision 0.8674	recall 0.8686	F1 0.8680
Comparison.Concession                     precision 0.6695	recall 0.2865	F1 0.4013
Comparison.Contrast                       precision 0.7519	recall 0.9260	F1 0.8299
Contingency.Cause.Reason                  precision 0.8899	recall 0.9371	F1 0.9129
Contingency.Cause.Result                  precision 0.9862	recall 0.9106	F1 0.9469
Contingency.Condition                     precision 0.9580	recall 0.8838	F1 0.9194
Expansion.Alternative                     precision 0.9511	recall 0.9162	F1 0.9333
Expansion.Alternative.Chosen alternative  precision 0.9592	recall 0.9691	F1 0.9641
Expansion.Conjunction                     precision 0.9565	recall 0.9650	F1 0.9608
Expansion.Exception                       precision 0.8889	recall 0.6154	F1 0.7273
Expansion.Instantiation                   precision 0.9872	recall 0.9788	F1 0.9830
Expansion.Restatement                     precision 0.7030	recall 0.5917	F1 0.6425
Temporal.Asynchronous.Precedence          precision 0.9216	recall 0.9817	F1 0.9507
Temporal.Asynchronous.Succession          precision 0.9667	recall 0.7622	F1 0.8524
Temporal.Synchrony                        precision 0.7027	recall 0.7961	F1 0.7465
Overall parser performance --------------
Precision 0.8674 Recall 0.8686 F1 0.8680

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg 2 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 0.9988 F1 0.9994
Sense classification--------------
*Micro-Average                            precision 0.4499	recall 0.4494	F1 0.4497
Comparison.Concession                     precision 0.8571	recall 0.0305	F1 0.0588
Comparison.Contrast                       precision 0.7862	recall 0.1386	F1 0.2357
Contingency.Cause.Reason                  precision 0.4628	recall 0.4039	F1 0.4314
Contingency.Cause.Result                  precision 0.6495	recall 0.2072	F1 0.3142
Contingency.Condition                     precision 1.0000	recall 0.7500	F1 0.8571
EntRel                                    precision 0.3987	recall 0.8938	F1 0.5515
Expansion.Alternative                     precision 1.0000	recall 0.8182	F1 0.9000
Expansion.Alternative.Chosen alternative  precision 0.9667	recall 0.2042	F1 0.3372
Expansion.Conjunction                     precision 0.4585	recall 0.4412	F1 0.4497
Expansion.Exception                       precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Instantiation                   precision 0.8394	recall 0.2565	F1 0.3929
Expansion.Restatement                     precision 0.4498	recall 0.3362	F1 0.3848
Temporal.Asynchronous.Precedence          precision 0.9367	recall 0.1616	F1 0.2756
Temporal.Asynchronous.Succession          precision 0.9688	recall 0.2183	F1 0.3563
Temporal.Synchrony                        precision 1.0000	recall 0.1314	F1 0.2323
Overall parser performance --------------
Precision 0.4499 Recall 0.4494 F1 0.4497
```

DEV SET (no explicit LSTM)
==========================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.7116	recall 0.7116	F1 0.7116
Comparison.Concession                     precision 0.5455	recall 0.3529	F1 0.4286
Comparison.Contrast                       precision 0.9309	recall 0.7000	F1 0.7991
Contingency.Cause.Reason                  precision 0.7025	recall 0.6489	F1 0.6746
Contingency.Cause.Result                  precision 0.6829	recall 0.3889	F1 0.4956
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
EntRel                                    precision 0.4758	recall 0.9581	F1 0.6358
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.8194	recall 0.7929	F1 0.8059
Expansion.Instantiation                   precision 1.0000	recall 0.3158	F1 0.4800
Expansion.Restatement                     precision 0.5059	recall 0.3945	F1 0.4433
Temporal.Asynchronous.Precedence          precision 0.9643	recall 0.7105	F1 0.8182
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.7451	F1 0.8539
Temporal.Synchrony                        precision 0.8000	recall 0.8000	F1 0.8000
Overall parser performance --------------
Precision 0.7116 Recall 0.7116 F1 0.7116

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9302	recall 0.9302	F1 0.9302
Comparison.Concession                     precision 0.5455	recall 0.5000	F1 0.5217
Comparison.Contrast                       precision 0.9518	recall 0.9634	F1 0.9576
Contingency.Cause.Reason                  precision 0.8333	recall 0.9259	F1 0.8772
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9628	recall 0.9837	F1 0.9731
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9600	recall 0.9796	F1 0.9697
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.7917	F1 0.8837
Temporal.Synchrony                        precision 0.7931	recall 0.8364	F1 0.8142
Overall parser performance --------------
Precision 0.9302 Recall 0.9302 F1 0.9302

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5194	recall 0.5194	F1 0.5194
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.7727	recall 0.1977	F1 0.3148
Contingency.Cause.Reason                  precision 0.5738	recall 0.4545	F1 0.5072
Contingency.Cause.Result                  precision 0.4800	recall 0.2264	F1 0.3077
EntRel                                    precision 0.4758	recall 0.9581	F1 0.6358
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.5766	recall 0.5120	F1 0.5424
Expansion.Instantiation                   precision 1.0000	recall 0.1875	F1 0.3158
Expansion.Restatement                     precision 0.4750	recall 0.3689	F1 0.4153
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.2222	F1 0.3636
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.4000	F1 0.5714
Overall parser performance --------------
Precision 0.5194 Recall 0.5194 F1 0.5194
```

TRAIN SET (no explicit LSTM)
============================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6390	recall 0.6390	F1 0.6390
Comparison.Concession                     precision 0.6709	recall 0.2455	F1 0.3595
Comparison.Contrast                       precision 0.7564	recall 0.6542	F1 0.7016
Contingency.Cause.Reason                  precision 0.6522	recall 0.5801	F1 0.6141
Contingency.Cause.Result                  precision 0.8220	recall 0.3759	F1 0.5159
Contingency.Condition                     precision 0.9581	recall 0.8833	F1 0.9192
EntRel                                    precision 0.3783	recall 0.9506	F1 0.5412
Expansion.Alternative                     precision 0.9529	recall 0.9010	F1 0.9262
Expansion.Alternative.Chosen alternative  precision 0.9600	recall 0.5021	F1 0.6593
Expansion.Conjunction                     precision 0.7895	recall 0.7136	F1 0.7496
Expansion.Exception                       precision 0.9000	recall 0.6000	F1 0.7200
Expansion.Instantiation                   precision 0.9202	recall 0.4206	F1 0.5773
Expansion.Restatement                     precision 0.5159	recall 0.3162	F1 0.3921
Temporal.Asynchronous.Precedence          precision 0.9256	recall 0.6708	F1 0.7778
Temporal.Asynchronous.Succession          precision 0.9686	recall 0.6612	F1 0.7859
Temporal.Synchrony                        precision 0.7057	recall 0.7117	F1 0.7087
Overall parser performance --------------
Precision 0.6390 Recall 0.6390 F1 0.6390

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 2 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg1 Arg2 extractor combined : Precision 0.9986 Recall 1.0000 F1 0.9993
Sense classification--------------
*Micro-Average                            precision 0.8674	recall 0.8686	F1 0.8680
Comparison.Concession                     precision 0.6695	recall 0.2865	F1 0.4013
Comparison.Contrast                       precision 0.7519	recall 0.9260	F1 0.8299
Contingency.Cause.Reason                  precision 0.8899	recall 0.9371	F1 0.9129
Contingency.Cause.Result                  precision 0.9862	recall 0.9106	F1 0.9469
Contingency.Condition                     precision 0.9580	recall 0.8838	F1 0.9194
Expansion.Alternative                     precision 0.9511	recall 0.9162	F1 0.9333
Expansion.Alternative.Chosen alternative  precision 0.9592	recall 0.9691	F1 0.9641
Expansion.Conjunction                     precision 0.9565	recall 0.9650	F1 0.9608
Expansion.Exception                       precision 0.8889	recall 0.6154	F1 0.7273
Expansion.Instantiation                   precision 0.9872	recall 0.9788	F1 0.9830
Expansion.Restatement                     precision 0.7030	recall 0.5917	F1 0.6425
Temporal.Asynchronous.Precedence          precision 0.9216	recall 0.9817	F1 0.9507
Temporal.Asynchronous.Succession          precision 0.9667	recall 0.7622	F1 0.8524
Temporal.Synchrony                        precision 0.7027	recall 0.7961	F1 0.7465
Overall parser performance --------------
Precision 0.8674 Recall 0.8686 F1 0.8680

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg 2 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 0.9988 F1 0.9994
Sense classification--------------
*Micro-Average                            precision 0.4515	recall 0.4509	F1 0.4512
Comparison.Concession                     precision 0.8000	recall 0.0203	F1 0.0396
Comparison.Contrast                       precision 0.8043	recall 0.1599	F1 0.2667
Contingency.Cause.Reason                  precision 0.5081	recall 0.4130	F1 0.4556
Contingency.Cause.Result                  precision 0.6723	recall 0.2105	F1 0.3206
Contingency.Condition                     precision 1.0000	recall 0.7500	F1 0.8571
EntRel                                    precision 0.3783	recall 0.9506	F1 0.5412
Expansion.Alternative                     precision 1.0000	recall 0.6364	F1 0.7778
Expansion.Alternative.Chosen alternative  precision 0.9630	recall 0.1831	F1 0.3077
Expansion.Conjunction                     precision 0.4981	recall 0.3818	F1 0.4322
Expansion.Exception                       precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Instantiation                   precision 0.8815	recall 0.3072	F1 0.4556
Expansion.Restatement                     precision 0.5036	recall 0.3032	F1 0.3785
Temporal.Asynchronous.Precedence          precision 0.9437	recall 0.1463	F1 0.2533
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.0629	F1 0.1184
Temporal.Synchrony                        precision 1.0000	recall 0.0863	F1 0.1589
Overall parser performance --------------
Precision 0.4515 Recall 0.4509 F1 0.4512
```

DEV SET (no SVD)
================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.7315	recall 0.7315	F1 0.7315
Comparison.Concession                     precision 0.8333	recall 0.2941	F1 0.4348
Comparison.Contrast                       precision 0.9275	recall 0.7160	F1 0.8081
Contingency.Cause.Reason                  precision 0.7542	recall 0.6899	F1 0.7206
Contingency.Cause.Result                  precision 0.8000	recall 0.3889	F1 0.5234
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
EntRel                                    precision 0.5037	recall 0.9395	F1 0.6558
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.7950	recall 0.8155	F1 0.8051
Expansion.Instantiation                   precision 1.0000	recall 0.3509	F1 0.5195
Expansion.Restatement                     precision 0.5165	recall 0.4312	F1 0.4700
Temporal.Asynchronous.Precedence          precision 0.9649	recall 0.7237	F1 0.8271
Temporal.Asynchronous.Succession          precision 0.9524	recall 0.7843	F1 0.8602
Temporal.Synchrony                        precision 0.8169	recall 0.9355	F1 0.8722
Overall parser performance --------------
Precision 0.7315 Recall 0.7315 F1 0.7315

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9484	recall 0.9484	F1 0.9484
Comparison.Concession                     precision 0.8333	recall 0.4167	F1 0.5556
Comparison.Contrast                       precision 0.9474	recall 0.9878	F1 0.9672
Contingency.Cause.Reason                  precision 0.9796	recall 0.9231	F1 0.9505
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9679	recall 0.9837	F1 0.9757
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.8125	F1 0.8966
Temporal.Synchrony                        precision 0.8116	recall 0.9825	F1 0.8889
Overall parser performance --------------
Precision 0.9484 Recall 0.9484 F1 0.9484

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5407	recall 0.5407	F1 0.5407
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.7727	recall 0.1977	F1 0.3148
Contingency.Cause.Reason                  precision 0.5942	recall 0.5325	F1 0.5616
Contingency.Cause.Result                  precision 0.6316	recall 0.2264	F1 0.3333
EntRel                                    precision 0.5037	recall 0.9395	F1 0.6558
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.5462	recall 0.5680	F1 0.5569
Expansion.Instantiation                   precision 1.0000	recall 0.2292	F1 0.3729
Expansion.Restatement                     precision 0.4884	recall 0.4078	F1 0.4444
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.2222	F1 0.3636
Temporal.Asynchronous.Succession          precision 0.3333	recall 0.3333	F1 0.3333
Temporal.Synchrony                        precision 1.0000	recall 0.4000	F1 0.5714
Overall parser performance --------------
Precision 0.5407 Recall 0.5407 F1 0.5407
```

TRAIN SET (no SVD)
==================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6563	recall 0.6563	F1 0.6563
Comparison.Concession                     precision 0.8324	recall 0.2220	F1 0.3505
Comparison.Contrast                       precision 0.7583	recall 0.6692	F1 0.7110
Contingency.Cause.Reason                  precision 0.6572	recall 0.6044	F1 0.6297
Contingency.Cause.Result                  precision 0.9232	recall 0.3864	F1 0.5448
Contingency.Condition                     precision 0.9581	recall 0.8841	F1 0.9196
EntRel                                    precision 0.4021	recall 0.9226	F1 0.5601
Expansion.Alternative                     precision 0.9485	recall 0.9154	F1 0.9316
Expansion.Alternative.Chosen alternative  precision 0.9612	recall 0.5167	F1 0.6721
Expansion.Conjunction                     precision 0.7712	recall 0.7425	F1 0.7566
Expansion.Exception                       precision 0.9000	recall 0.6000	F1 0.7200
Expansion.Instantiation                   precision 0.9193	recall 0.4642	F1 0.6169
Expansion.Restatement                     precision 0.5024	recall 0.3597	F1 0.4192
Temporal.Asynchronous.Precedence          precision 0.9291	recall 0.6754	F1 0.7822
Temporal.Asynchronous.Succession          precision 0.9711	recall 0.6871	F1 0.8048
Temporal.Synchrony                        precision 0.7226	recall 0.7868	F1 0.7533
Overall parser performance --------------
Precision 0.6563 Recall 0.6563 F1 0.6563

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 2 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg1 Arg2 extractor combined : Precision 0.9986 Recall 1.0000 F1 0.9993
Sense classification--------------
*Micro-Average                            precision 0.8797	recall 0.8810	F1 0.8803
Comparison.Concession                     precision 0.8299	recall 0.2579	F1 0.3935
Comparison.Contrast                       precision 0.7549	recall 0.9605	F1 0.8454
Contingency.Cause.Reason                  precision 0.9573	recall 0.9312	F1 0.9441
Contingency.Cause.Result                  precision 0.9930	recall 0.9106	F1 0.9501
Contingency.Condition                     precision 0.9580	recall 0.8846	F1 0.9198
Expansion.Alternative                     precision 0.9462	recall 0.9263	F1 0.9362
Expansion.Alternative.Chosen alternative  precision 0.9600	recall 0.9796	F1 0.9697
Expansion.Conjunction                     precision 0.9613	recall 0.9657	F1 0.9635
Expansion.Exception                       precision 0.8889	recall 0.6154	F1 0.7273
Expansion.Instantiation                   precision 0.9914	recall 0.9788	F1 0.9851
Expansion.Restatement                     precision 0.7522	recall 0.7083	F1 0.7296
Temporal.Asynchronous.Precedence          precision 0.9237	recall 0.9817	F1 0.9518
Temporal.Asynchronous.Succession          precision 0.9758	recall 0.7727	F1 0.8625
Temporal.Synchrony                        precision 0.7191	recall 0.8730	F1 0.7886
Overall parser performance --------------
Precision 0.8797 Recall 0.8810 F1 0.8803

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg 2 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 0.9988 F1 0.9994
Sense classification--------------
*Micro-Average                            precision 0.4728	recall 0.4722	F1 0.4725
Comparison.Concession                     precision 1.0000	recall 0.0254	F1 0.0495
Comparison.Contrast                       precision 0.8007	recall 0.1391	F1 0.2371
Contingency.Cause.Reason                  precision 0.5065	recall 0.4535	F1 0.4785
Contingency.Cause.Result                  precision 0.8483	recall 0.2243	F1 0.3548
Contingency.Condition                     precision 1.0000	recall 0.7500	F1 0.8571
EntRel                                    precision 0.4021	recall 0.9226	F1 0.5601
Expansion.Alternative                     precision 1.0000	recall 0.7273	F1 0.8421
Expansion.Alternative.Chosen alternative  precision 0.9655	recall 0.1972	F1 0.3275
Expansion.Conjunction                     precision 0.4931	recall 0.4480	F1 0.4694
Expansion.Exception                       precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Instantiation                   precision 0.8837	recall 0.3597	F1 0.5113
Expansion.Restatement                     precision 0.4866	recall 0.3433	F1 0.4026
Temporal.Asynchronous.Precedence          precision 0.9605	recall 0.1594	F1 0.2734
Temporal.Asynchronous.Succession          precision 0.8333	recall 0.1761	F1 0.2907
Temporal.Synchrony                        precision 1.0000	recall 0.1185	F1 0.2119
Overall parser performance --------------
Precision 0.4728 Recall 0.4722 F1 0.4725
```

DEV SET (no explicit SVD)
=========================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.7209	recall 0.7209	F1 0.7209
Comparison.Concession                     precision 0.8333	recall 0.2941	F1 0.4348
Comparison.Contrast                       precision 0.9275	recall 0.7160	F1 0.8081
Contingency.Cause.Reason                  precision 0.7568	recall 0.6512	F1 0.7000
Contingency.Cause.Result                  precision 0.8000	recall 0.3889	F1 0.5234
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
EntRel                                    precision 0.4725	recall 0.9581	F1 0.6329
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.8221	recall 0.7929	F1 0.8072
Expansion.Instantiation                   precision 1.0000	recall 0.3158	F1 0.4800
Expansion.Restatement                     precision 0.5059	recall 0.3945	F1 0.4433
Temporal.Asynchronous.Precedence          precision 0.9649	recall 0.7237	F1 0.8271
Temporal.Asynchronous.Succession          precision 0.9512	recall 0.7647	F1 0.8478
Temporal.Synchrony                        precision 0.8169	recall 0.9355	F1 0.8722
Overall parser performance --------------
Precision 0.7209 Recall 0.7209 F1 0.7209

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9484	recall 0.9484	F1 0.9484
Comparison.Concession                     precision 0.8333	recall 0.4167	F1 0.5556
Comparison.Contrast                       precision 0.9474	recall 0.9878	F1 0.9672
Contingency.Cause.Reason                  precision 0.9796	recall 0.9231	F1 0.9505
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9778	recall 0.9362	F1 0.9565
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9679	recall 0.9837	F1 0.9757
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.8125	F1 0.8966
Temporal.Synchrony                        precision 0.8116	recall 0.9825	F1 0.8889
Overall parser performance --------------
Precision 0.9484 Recall 0.9484 F1 0.9484

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5207	recall 0.5207	F1 0.5207
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.7727	recall 0.1977	F1 0.3148
Contingency.Cause.Reason                  precision 0.5806	recall 0.4675	F1 0.5180
Contingency.Cause.Result                  precision 0.6316	recall 0.2264	F1 0.3333
EntRel                                    precision 0.4725	recall 0.9581	F1 0.6329
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.5766	recall 0.5120	F1 0.5424
Expansion.Instantiation                   precision 1.0000	recall 0.1875	F1 0.3158
Expansion.Restatement                     precision 0.4750	recall 0.3689	F1 0.4153
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.2222	F1 0.3636
Temporal.Asynchronous.Succession          precision 0.0000	recall 0.0000	F1 0.0000
Temporal.Synchrony                        precision 1.0000	recall 0.4000	F1 0.5714
Overall parser performance --------------
Precision 0.5207 Recall 0.5207 F1 0.5207
```

TRAIN SET (no explicit SVD)
===========================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6451	recall 0.6451	F1 0.6451
Comparison.Concession                     precision 0.8319	recall 0.2212	F1 0.3494
Comparison.Contrast                       precision 0.7590	recall 0.6765	F1 0.7154
Contingency.Cause.Reason                  precision 0.6667	recall 0.5784	F1 0.6194
Contingency.Cause.Result                  precision 0.9292	recall 0.3759	F1 0.5352
Contingency.Condition                     precision 0.9581	recall 0.8841	F1 0.9196
EntRel                                    precision 0.3738	recall 0.9506	F1 0.5366
Expansion.Alternative                     precision 0.9482	recall 0.9104	F1 0.9289
Expansion.Alternative.Chosen alternative  precision 0.9606	recall 0.5083	F1 0.6649
Expansion.Conjunction                     precision 0.7982	recall 0.7139	F1 0.7537
Expansion.Exception                       precision 0.9000	recall 0.6000	F1 0.7200
Expansion.Instantiation                   precision 0.9202	recall 0.4206	F1 0.5773
Expansion.Restatement                     precision 0.5223	recall 0.3248	F1 0.4006
Temporal.Asynchronous.Precedence          precision 0.9287	recall 0.6705	F1 0.7787
Temporal.Asynchronous.Succession          precision 0.9706	recall 0.6742	F1 0.7957
Temporal.Synchrony                        precision 0.7217	recall 0.7808	F1 0.7501
Overall parser performance --------------
Precision 0.6451 Recall 0.6451 F1 0.6451

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 2 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg1 Arg2 extractor combined : Precision 0.9986 Recall 1.0000 F1 0.9993
Sense classification--------------
*Micro-Average                            precision 0.8797	recall 0.8810	F1 0.8803
Comparison.Concession                     precision 0.8299	recall 0.2579	F1 0.3935
Comparison.Contrast                       precision 0.7549	recall 0.9605	F1 0.8454
Contingency.Cause.Reason                  precision 0.9573	recall 0.9312	F1 0.9441
Contingency.Cause.Result                  precision 0.9930	recall 0.9106	F1 0.9501
Contingency.Condition                     precision 0.9580	recall 0.8846	F1 0.9198
Expansion.Alternative                     precision 0.9462	recall 0.9263	F1 0.9362
Expansion.Alternative.Chosen alternative  precision 0.9600	recall 0.9796	F1 0.9697
Expansion.Conjunction                     precision 0.9613	recall 0.9657	F1 0.9635
Expansion.Exception                       precision 0.8889	recall 0.6154	F1 0.7273
Expansion.Instantiation                   precision 0.9914	recall 0.9788	F1 0.9851
Expansion.Restatement                     precision 0.7522	recall 0.7083	F1 0.7296
Temporal.Asynchronous.Precedence          precision 0.9237	recall 0.9817	F1 0.9518
Temporal.Asynchronous.Succession          precision 0.9758	recall 0.7727	F1 0.8625
Temporal.Synchrony                        precision 0.7191	recall 0.8730	F1 0.7886
Overall parser performance --------------
Precision 0.8797 Recall 0.8810 F1 0.8803

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg 2 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 0.9988 F1 0.9994
Sense classification--------------
*Micro-Average                            precision 0.4524	recall 0.4519	F1 0.4522
Comparison.Concession                     precision 1.0000	recall 0.0203	F1 0.0398
Comparison.Contrast                       precision 0.8043	recall 0.1598	F1 0.2666
Contingency.Cause.Reason                  precision 0.5071	recall 0.4153	F1 0.4567
Contingency.Cause.Result                  precision 0.8556	recall 0.2105	F1 0.3379
Contingency.Condition                     precision 1.0000	recall 0.7500	F1 0.8571
EntRel                                    precision 0.3738	recall 0.9506	F1 0.5366
Expansion.Alternative                     precision 1.0000	recall 0.6364	F1 0.7778
Expansion.Alternative.Chosen alternative  precision 0.9630	recall 0.1831	F1 0.3077
Expansion.Conjunction                     precision 0.5086	recall 0.3816	F1 0.4360
Expansion.Exception                       precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Instantiation                   precision 0.8793	recall 0.3072	F1 0.4554
Expansion.Restatement                     precision 0.5055	recall 0.3068	F1 0.3818
Temporal.Asynchronous.Precedence          precision 0.9571	recall 0.1463	F1 0.2538
Temporal.Asynchronous.Succession          precision 0.7222	recall 0.0909	F1 0.1615
Temporal.Synchrony                        precision 1.0000	recall 0.0863	F1 0.1589
Overall parser performance --------------
Precision 0.4524 Recall 0.4519 F1 0.4522
```

DEV SET (no SVD, no MAJOR)
==========================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.7429	recall 0.7429	F1 0.7429
Comparison.Concession                     precision 0.8333	recall 0.2941	F1 0.4348
Comparison.Contrast                       precision 0.9275	recall 0.7160	F1 0.8081
Contingency.Cause.Reason                  precision 0.7500	recall 0.7099	F1 0.7294
Contingency.Cause.Result                  precision 0.7073	recall 0.4028	F1 0.5133
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
EntRel                                    precision 0.5262	recall 0.9349	F1 0.6734
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.6250	F1 0.7692
Expansion.Conjunction                     precision 0.7975	recall 0.8155	F1 0.8064
Expansion.Instantiation                   precision 1.0000	recall 0.4737	F1 0.6429
Expansion.Restatement                     precision 0.5474	recall 0.4771	F1 0.5098
Temporal.Asynchronous.Precedence          precision 0.9655	recall 0.7368	F1 0.8358
Temporal.Asynchronous.Succession          precision 0.9535	recall 0.8039	F1 0.8723
Temporal.Synchrony                        precision 0.8462	recall 0.9167	F1 0.8800
Overall parser performance --------------
Precision 0.7429 Recall 0.7429 F1 0.7429

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9530	recall 0.9530	F1 0.9530
Comparison.Concession                     precision 0.8333	recall 0.4167	F1 0.5556
Comparison.Contrast                       precision 0.9474	recall 0.9878	F1 0.9672
Contingency.Cause.Reason                  precision 0.9623	recall 0.9444	F1 0.9533
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9731	recall 0.9837	F1 0.9784
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 1.0000	F1 1.0000
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Synchrony                        precision 0.8413	recall 0.9636	F1 0.8983
Overall parser performance --------------
Precision 0.9530 Recall 0.9530 F1 0.9530

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.5581	recall 0.5581	F1 0.5581
Comparison.Concession                     precision 1.0000	recall 0.0000	F1 0.0000
Comparison.Contrast                       precision 0.7727	recall 0.1977	F1 0.3148
Contingency.Cause.Reason                  precision 0.5915	recall 0.5455	F1 0.5676
Contingency.Cause.Result                  precision 0.5200	recall 0.2453	F1 0.3333
EntRel                                    precision 0.5262	recall 0.9349	F1 0.6734
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.0000	F1 0.0000
Expansion.Conjunction                     precision 0.5462	recall 0.5680	F1 0.5569
Expansion.Instantiation                   precision 1.0000	recall 0.3750	F1 0.5455
Expansion.Restatement                     precision 0.5169	recall 0.4466	F1 0.4792
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.2593	F1 0.4118
Temporal.Asynchronous.Succession          precision 0.3333	recall 0.3333	F1 0.3333
Temporal.Synchrony                        precision 1.0000	recall 0.4000	F1 0.5714
Overall parser performance --------------
Precision 0.5581 Recall 0.5581 F1 0.5581
```

TRAIN SET (no SVD, no MAJOR)
============================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.6728	recall 0.6728	F1 0.6728
Comparison.Concession                     precision 0.8739	recall 0.2277	F1 0.3613
Comparison.Contrast                       precision 0.7660	recall 0.6794	F1 0.7201
Contingency.Cause.Reason                  precision 0.6548	recall 0.6398	F1 0.6472
Contingency.Cause.Result                  precision 0.8401	recall 0.4040	F1 0.5456
Contingency.Condition                     precision 0.9639	recall 0.9132	F1 0.9379
EntRel                                    precision 0.4273	recall 0.9173	F1 0.5830
Expansion.Alternative                     precision 0.9541	recall 0.9257	F1 0.9397
Expansion.Alternative.Chosen alternative  precision 0.9615	recall 0.5230	F1 0.6775
Expansion.Conjunction                     precision 0.7709	recall 0.7461	F1 0.7583
Expansion.Exception                       precision 0.9000	recall 0.6000	F1 0.7200
Expansion.Instantiation                   precision 0.9042	recall 0.5808	F1 0.7073
Expansion.Restatement                     precision 0.5090	recall 0.3710	F1 0.4292
Temporal.Asynchronous.Precedence          precision 0.9327	recall 0.6909	F1 0.7938
Temporal.Asynchronous.Succession          precision 0.9735	recall 0.7518	F1 0.8484
Temporal.Synchrony                        precision 0.7806	recall 0.7974	F1 0.7889
Overall parser performance --------------
Precision 0.6728 Recall 0.6728 F1 0.6728

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 2 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg1 Arg2 extractor combined : Precision 0.9986 Recall 1.0000 F1 0.9993
Sense classification--------------
*Micro-Average                            precision 0.8902	recall 0.8915	F1 0.8909
Comparison.Concession                     precision 0.8742	recall 0.2636	F1 0.4051
Comparison.Contrast                       precision 0.7618	recall 0.9639	F1 0.8510
Contingency.Cause.Reason                  precision 0.9413	recall 0.9592	F1 0.9502
Contingency.Cause.Result                  precision 0.9931	recall 0.9149	F1 0.9524
Contingency.Condition                     precision 0.9638	recall 0.9138	F1 0.9381
Expansion.Alternative                     precision 0.9519	recall 0.9319	F1 0.9418
Expansion.Alternative.Chosen alternative  precision 0.9596	recall 0.9794	F1 0.9694
Expansion.Conjunction                     precision 0.9620	recall 0.9657	F1 0.9639
Expansion.Exception                       precision 0.8889	recall 0.6154	F1 0.7273
Expansion.Instantiation                   precision 0.9872	recall 0.9788	F1 0.9830
Expansion.Restatement                     precision 0.7627	recall 0.7500	F1 0.7563
Temporal.Asynchronous.Precedence          precision 0.9295	recall 0.9830	F1 0.9555
Temporal.Asynchronous.Succession          precision 0.9790	recall 0.8363	F1 0.9021
Temporal.Synchrony                        precision 0.7846	recall 0.8849	F1 0.8317
Overall parser performance --------------
Precision 0.8902 Recall 0.8915 F1 0.8909

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg 2 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 0.9988 F1 0.9994
Sense classification--------------
*Micro-Average                            precision 0.4942	recall 0.4936	F1 0.4939
Comparison.Concession                     precision 0.8571	recall 0.0305	F1 0.0588
Comparison.Contrast                       precision 0.8110	recall 0.1617	F1 0.2696
Contingency.Cause.Reason                  precision 0.5117	recall 0.4899	F1 0.5006
Contingency.Cause.Result                  precision 0.7137	recall 0.2461	F1 0.3659
Contingency.Condition                     precision 1.0000	recall 0.7500	F1 0.8571
EntRel                                    precision 0.4273	recall 0.9173	F1 0.5830
Expansion.Alternative                     precision 1.0000	recall 0.8182	F1 0.9000
Expansion.Alternative.Chosen alternative  precision 0.9677	recall 0.2113	F1 0.3468
Expansion.Conjunction                     precision 0.4955	recall 0.4562	F1 0.4750
Expansion.Exception                       precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Instantiation                   precision 0.8750	recall 0.5000	F1 0.6364
Expansion.Restatement                     precision 0.4926	recall 0.3531	F1 0.4113
Temporal.Asynchronous.Precedence          precision 0.9381	recall 0.1987	F1 0.3279
Temporal.Asynchronous.Succession          precision 0.8537	recall 0.2465	F1 0.3825
Temporal.Synchrony                        precision 0.6207	recall 0.1333	F1 0.2195
Overall parser performance --------------
Precision 0.4942 Recall 0.4936 F1 0.4939
```

LAST RUNS ORIG (DEV SET: Wang, XGBoost, LSTM)
=============================================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9162	recall 0.9162	F1 0.9162
Comparison.Concession                     precision 0.7500	recall 0.5294	F1 0.6207
Comparison.Contrast                       precision 0.9592	recall 0.9325	F1 0.9457
Contingency.Cause.Reason                  precision 0.9179	recall 0.9248	F1 0.9213
Contingency.Cause.Result                  precision 0.9841	recall 0.8378	F1 0.9051
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
EntRel                                    precision 0.7852	recall 0.9860	F1 0.8742
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.7500	F1 0.8571
Expansion.Conjunction                     precision 0.9597	recall 0.9286	F1 0.9439
Expansion.Instantiation                   precision 1.0000	recall 0.8596	F1 0.9245
Expansion.Restatement                     precision 0.9175	recall 0.8241	F1 0.8683
Temporal.Asynchronous.Precedence          precision 0.9718	recall 0.9324	F1 0.9517
Temporal.Asynchronous.Succession          precision 0.9773	recall 0.8431	F1 0.9053
Temporal.Synchrony                        precision 0.8485	recall 0.9655	F1 0.9032
Overall parser performance --------------
Precision 0.9162 Recall 0.9162 F1 0.9162

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9514	recall 0.9514	F1 0.9514
Comparison.Concession                     precision 0.6667	recall 0.5000	F1 0.5714
Comparison.Contrast                       precision 0.9524	recall 0.9756	F1 0.9639
Contingency.Cause.Reason                  precision 0.9808	recall 0.9273	F1 0.9533
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9679	recall 0.9837	F1 0.9757
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.8542	F1 0.9213
Temporal.Synchrony                        precision 0.8413	recall 0.9815	F1 0.9060
Overall parser performance --------------
Precision 0.9514 Recall 0.9514 F1 0.9514

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.8852	recall 0.8852	F1 0.8852
Comparison.Concession                     precision 1.0000	recall 0.6000	F1 0.7500
Comparison.Contrast                       precision 0.9740	recall 0.8523	F1 0.9091
Contingency.Cause.Reason                  precision 0.8780	recall 0.9231	F1 0.9000
Contingency.Cause.Result                  precision 0.9787	recall 0.8364	F1 0.9020
EntRel                                    precision 0.7852	recall 0.9860	F1 0.8742
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Conjunction                     precision 0.9459	recall 0.8468	F1 0.8936
Expansion.Instantiation                   precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Restatement                     precision 0.9130	recall 0.8235	F1 0.8660
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.8000	F1 0.8889
Temporal.Asynchronous.Succession          precision 0.6667	recall 0.6667	F1 0.6667
Temporal.Synchrony                        precision 1.0000	recall 0.7500	F1 0.8571
Overall parser performance --------------
Precision 0.8852 Recall 0.8852 F1 0.8852
```

LAST RUNS ORIG (TRAIN SET: Wang, XGBoost, LSTM)
===============================================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.8637	recall 0.8637	F1 0.8637
Comparison.Concession                     precision 0.8008	recall 0.3273	F1 0.4647
Comparison.Contrast                       precision 0.8211	recall 0.8735	F1 0.8465
Contingency.Cause.Reason                  precision 0.8852	recall 0.9075	F1 0.8962
Contingency.Cause.Result                  precision 0.9826	recall 0.8473	F1 0.9099
Contingency.Condition                     precision 0.9649	recall 0.9289	F1 0.9465
EntRel                                    precision 0.6700	recall 0.9823	F1 0.7966
Expansion.Alternative                     precision 0.9550	recall 0.9455	F1 0.9502
Expansion.Alternative.Chosen alternative  precision 0.9751	recall 0.8201	F1 0.8909
Expansion.Conjunction                     precision 0.9431	recall 0.8821	F1 0.9116
Expansion.Exception                       precision 0.9091	recall 0.6667	F1 0.7692
Expansion.Instantiation                   precision 0.9725	recall 0.8887	F1 0.9287
Expansion.Restatement                     precision 0.9258	recall 0.7944	F1 0.8551
Temporal.Asynchronous.Precedence          precision 0.9540	recall 0.8881	F1 0.9199
Temporal.Asynchronous.Succession          precision 0.9836	recall 0.8046	F1 0.8851
Temporal.Synchrony                        precision 0.8129	recall 0.8634	F1 0.8373
Overall parser performance --------------
Precision 0.8637 Recall 0.8637 F1 0.8637

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 2 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg1 Arg2 extractor combined : Precision 0.9986 Recall 1.0000 F1 0.9993
Sense classification--------------
*Micro-Average                            precision 0.8931	recall 0.8944	F1 0.8938
Comparison.Concession                     precision 0.7564	recall 0.2988	F1 0.4284
Comparison.Contrast                       precision 0.7644	recall 0.9458	F1 0.8455
Contingency.Cause.Reason                  precision 0.9674	recall 0.9655	F1 0.9664
Contingency.Cause.Result                  precision 0.9931	recall 0.9153	F1 0.9526
Contingency.Condition                     precision 0.9647	recall 0.9286	F1 0.9463
Expansion.Alternative                     precision 0.9529	recall 0.9529	F1 0.9529
Expansion.Alternative.Chosen alternative  precision 0.9596	recall 0.9794	F1 0.9694
Expansion.Conjunction                     precision 0.9636	recall 0.9683	F1 0.9659
Expansion.Exception                       precision 0.8889	recall 0.6154	F1 0.7273
Expansion.Instantiation                   precision 0.9914	recall 0.9788	F1 0.9851
Expansion.Restatement                     precision 0.8333	recall 0.7025	F1 0.7623
Temporal.Asynchronous.Precedence          precision 0.9318	recall 0.9856	F1 0.9580
Temporal.Asynchronous.Succession          precision 0.9817	recall 0.8408	F1 0.9058
Temporal.Synchrony                        precision 0.8032	recall 0.9028	F1 0.8501
Overall parser performance --------------
Precision 0.8931 Recall 0.8944 F1 0.8938

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg 2 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 0.9988 F1 0.9994
Sense classification--------------
*Micro-Average                            precision 0.8386	recall 0.8376	F1 0.8381
Comparison.Concession                     precision 1.0000	recall 0.4847	F1 0.6529
Comparison.Contrast                       precision 0.9880	recall 0.7428	F1 0.8480
Contingency.Cause.Reason                  precision 0.8480	recall 0.8803	F1 0.8638
Contingency.Cause.Result                  precision 0.9790	recall 0.8262	F1 0.8962
Contingency.Condition                     precision 1.0000	recall 1.0000	F1 1.0000
EntRel                                    precision 0.6700	recall 0.9823	F1 0.7966
Expansion.Alternative                     precision 1.0000	recall 0.8182	F1 0.9000
Expansion.Alternative.Chosen alternative  precision 0.9902	recall 0.7113	F1 0.8279
Expansion.Conjunction                     precision 0.9069	recall 0.7662	F1 0.8306
Expansion.Exception                       precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Instantiation                   precision 0.9683	recall 0.8704	F1 0.9167
Expansion.Restatement                     precision 0.9302	recall 0.7988	F1 0.8595
Temporal.Asynchronous.Precedence          precision 0.9938	recall 0.7124	F1 0.8299
Temporal.Asynchronous.Succession          precision 0.9877	recall 0.5797	F1 0.7306
Temporal.Synchrony                        precision 0.9844	recall 0.5294	F1 0.6885
Overall parser performance --------------
Precision 0.8386 Recall 0.8376 F1 0.8381
```

LAST RUNS RECAST (DEV SET: Wang, XGBoost, LSTM)
===============================================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9162	recall 0.9162	F1 0.9162
Comparison.Concession                     precision 0.7500	recall 0.5294	F1 0.6207
Comparison.Contrast                       precision 0.9592	recall 0.9325	F1 0.9457
Contingency.Cause.Reason                  precision 0.9179	recall 0.9248	F1 0.9213
Contingency.Cause.Result                  precision 0.9841	recall 0.8378	F1 0.9051
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
EntRel                                    precision 0.7852	recall 0.9860	F1 0.8742
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.7500	F1 0.8571
Expansion.Conjunction                     precision 0.9597	recall 0.9286	F1 0.9439
Expansion.Instantiation                   precision 1.0000	recall 0.8596	F1 0.9245
Expansion.Restatement                     precision 0.9175	recall 0.8241	F1 0.8683
Temporal.Asynchronous.Precedence          precision 0.9718	recall 0.9324	F1 0.9517
Temporal.Asynchronous.Succession          precision 0.9773	recall 0.8431	F1 0.9053
Temporal.Synchrony                        precision 0.8485	recall 0.9655	F1 0.9032
Overall parser performance --------------
Precision 0.9162 Recall 0.9162 F1 0.9162

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9514	recall 0.9514	F1 0.9514
Comparison.Concession                     precision 0.6667	recall 0.5000	F1 0.5714
Comparison.Contrast                       precision 0.9524	recall 0.9756	F1 0.9639
Contingency.Cause.Reason                  precision 0.9808	recall 0.9273	F1 0.9533
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9679	recall 0.9837	F1 0.9757
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.8542	F1 0.9213
Temporal.Synchrony                        precision 0.8413	recall 0.9815	F1 0.9060
Overall parser performance --------------
Precision 0.9514 Recall 0.9514 F1 0.9514

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.8852	recall 0.8852	F1 0.8852
Comparison.Concession                     precision 1.0000	recall 0.6000	F1 0.7500
Comparison.Contrast                       precision 0.9740	recall 0.8523	F1 0.9091
Contingency.Cause.Reason                  precision 0.8780	recall 0.9231	F1 0.9000
Contingency.Cause.Result                  precision 0.9787	recall 0.8364	F1 0.9020
EntRel                                    precision 0.7852	recall 0.9860	F1 0.8742
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Conjunction                     precision 0.9459	recall 0.8468	F1 0.8936
Expansion.Instantiation                   precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Restatement                     precision 0.9130	recall 0.8235	F1 0.8660
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.8000	F1 0.8889
Temporal.Asynchronous.Succession          precision 0.6667	recall 0.6667	F1 0.6667
Temporal.Synchrony                        precision 1.0000	recall 0.7500	F1 0.8571
Overall parser performance --------------
Precision 0.8852 Recall 0.8852 F1 0.8852
```

LAST RUNS RECAST (TRAIN SET: Wang, XGBoost, LSTM)
=================================================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.8637	recall 0.8637	F1 0.8637
Comparison.Concession                     precision 0.8008	recall 0.3273	F1 0.4647
Comparison.Contrast                       precision 0.8211	recall 0.8735	F1 0.8465
Contingency.Cause.Reason                  precision 0.8852	recall 0.9075	F1 0.8962
Contingency.Cause.Result                  precision 0.9826	recall 0.8473	F1 0.9099
Contingency.Condition                     precision 0.9649	recall 0.9289	F1 0.9465
EntRel                                    precision 0.6700	recall 0.9823	F1 0.7966
Expansion.Alternative                     precision 0.9550	recall 0.9455	F1 0.9502
Expansion.Alternative.Chosen alternative  precision 0.9751	recall 0.8201	F1 0.8909
Expansion.Conjunction                     precision 0.9431	recall 0.8821	F1 0.9116
Expansion.Exception                       precision 0.9091	recall 0.6667	F1 0.7692
Expansion.Instantiation                   precision 0.9725	recall 0.8887	F1 0.9287
Expansion.Restatement                     precision 0.9258	recall 0.7944	F1 0.8551
Temporal.Asynchronous.Precedence          precision 0.9540	recall 0.8881	F1 0.9199
Temporal.Asynchronous.Succession          precision 0.9836	recall 0.8046	F1 0.8851
Temporal.Synchrony                        precision 0.8129	recall 0.8634	F1 0.8373
Overall parser performance --------------
Precision 0.8637 Recall 0.8637 F1 0.8637

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 1 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg 2 extractor              : Precision 0.9986 Recall 1.0000 F1 0.9993
Arg1 Arg2 extractor combined : Precision 0.9986 Recall 1.0000 F1 0.9993
Sense classification--------------
*Micro-Average                            precision 0.8931	recall 0.8944	F1 0.8938
Comparison.Concession                     precision 0.7564	recall 0.2988	F1 0.4284
Comparison.Contrast                       precision 0.7644	recall 0.9458	F1 0.8455
Contingency.Cause.Reason                  precision 0.9674	recall 0.9655	F1 0.9664
Contingency.Cause.Result                  precision 0.9931	recall 0.9153	F1 0.9526
Contingency.Condition                     precision 0.9647	recall 0.9286	F1 0.9463
Expansion.Alternative                     precision 0.9529	recall 0.9529	F1 0.9529
Expansion.Alternative.Chosen alternative  precision 0.9596	recall 0.9794	F1 0.9694
Expansion.Conjunction                     precision 0.9636	recall 0.9683	F1 0.9659
Expansion.Exception                       precision 0.8889	recall 0.6154	F1 0.7273
Expansion.Instantiation                   precision 0.9914	recall 0.9788	F1 0.9851
Expansion.Restatement                     precision 0.8333	recall 0.7025	F1 0.7623
Temporal.Asynchronous.Precedence          precision 0.9318	recall 0.9856	F1 0.9580
Temporal.Asynchronous.Succession          precision 0.9817	recall 0.8408	F1 0.9058
Temporal.Synchrony                        precision 0.8032	recall 0.9028	F1 0.8501
Overall parser performance --------------
Precision 0.8931 Recall 0.8944 F1 0.8938

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg 2 extractor              : Precision 1.0000 Recall 0.9988 F1 0.9994
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 0.9988 F1 0.9994
Sense classification--------------
*Micro-Average                            precision 0.8386	recall 0.8376	F1 0.8381
Comparison.Concession                     precision 1.0000	recall 0.4847	F1 0.6529
Comparison.Contrast                       precision 0.9880	recall 0.7428	F1 0.8480
Contingency.Cause.Reason                  precision 0.8480	recall 0.8803	F1 0.8638
Contingency.Cause.Result                  precision 0.9790	recall 0.8262	F1 0.8962
Contingency.Condition                     precision 1.0000	recall 1.0000	F1 1.0000
EntRel                                    precision 0.6700	recall 0.9823	F1 0.7966
Expansion.Alternative                     precision 1.0000	recall 0.8182	F1 0.9000
Expansion.Alternative.Chosen alternative  precision 0.9902	recall 0.7113	F1 0.8279
Expansion.Conjunction                     precision 0.9069	recall 0.7662	F1 0.8306
Expansion.Exception                       precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Instantiation                   precision 0.9683	recall 0.8704	F1 0.9167
Expansion.Restatement                     precision 0.9302	recall 0.7988	F1 0.8595
Temporal.Asynchronous.Precedence          precision 0.9938	recall 0.7124	F1 0.8299
Temporal.Asynchronous.Succession          precision 0.9877	recall 0.5797	F1 0.7306
Temporal.Synchrony                        precision 0.9844	recall 0.5294	F1 0.6885
Overall parser performance --------------
Precision 0.8386 Recall 0.8376 F1 0.8381
```

LAST RUNS RECAST CPU (DEV SET: Wang, XGBoost, LSTM)
===================================================

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9162	recall 0.9162	F1 0.9162
Comparison.Concession                     precision 0.7500	recall 0.5294	F1 0.6207
Comparison.Contrast                       precision 0.9592	recall 0.9325	F1 0.9457
Contingency.Cause.Reason                  precision 0.9179	recall 0.9248	F1 0.9213
Contingency.Cause.Result                  precision 0.9841	recall 0.8378	F1 0.9051
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
EntRel                                    precision 0.7852	recall 0.9860	F1 0.8742
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.7500	F1 0.8571
Expansion.Conjunction                     precision 0.9597	recall 0.9286	F1 0.9439
Expansion.Instantiation                   precision 1.0000	recall 0.8596	F1 0.9245
Expansion.Restatement                     precision 0.9175	recall 0.8241	F1 0.8683
Temporal.Asynchronous.Precedence          precision 0.9718	recall 0.9324	F1 0.9517
Temporal.Asynchronous.Succession          precision 0.9773	recall 0.8431	F1 0.9053
Temporal.Synchrony                        precision 0.8485	recall 0.9655	F1 0.9032
Overall parser performance --------------
Precision 0.9162 Recall 0.9162 F1 0.9162

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9514	recall 0.9514	F1 0.9514
Comparison.Concession                     precision 0.6667	recall 0.5000	F1 0.5714
Comparison.Contrast                       precision 0.9524	recall 0.9756	F1 0.9639
Contingency.Cause.Reason                  precision 0.9808	recall 0.9273	F1 0.9533
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9679	recall 0.9837	F1 0.9757
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.8542	F1 0.9213
Temporal.Synchrony                        precision 0.8413	recall 0.9815	F1 0.9060
Overall parser performance --------------
Precision 0.9514 Recall 0.9514 F1 0.9514

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.8852	recall 0.8852	F1 0.8852
Comparison.Concession                     precision 1.0000	recall 0.6000	F1 0.7500
Comparison.Contrast                       precision 0.9740	recall 0.8523	F1 0.9091
Contingency.Cause.Reason                  precision 0.8780	recall 0.9231	F1 0.9000
Contingency.Cause.Result                  precision 0.9787	recall 0.8364	F1 0.9020
EntRel                                    precision 0.7852	recall 0.9860	F1 0.8742
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Conjunction                     precision 0.9459	recall 0.8468	F1 0.8936
Expansion.Instantiation                   precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Restatement                     precision 0.9130	recall 0.8235	F1 0.8660
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.8000	F1 0.8889
Temporal.Asynchronous.Succession          precision 0.6667	recall 0.6667	F1 0.6667
Temporal.Synchrony                        precision 1.0000	recall 0.7500	F1 0.8571
Overall parser performance --------------
Precision 0.8852 Recall 0.8852 F1 0.8852
```

==================================================================

```
3f777a9786673ba01387ba65aa0b013c  dsenser/data/models/pdtb.sense.model.LSTMSenser
c852915080e526141beeec0fd02d9085  dsenser/data/models/pdtb.sense.model.SVDSenser
```

```
================================================
Evaluation for all discourse relations
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9162	recall 0.9162	F1 0.9162
Comparison.Concession                     precision 0.7500	recall 0.5294	F1 0.6207
Comparison.Contrast                       precision 0.9592	recall 0.9325	F1 0.9457
Contingency.Cause.Reason                  precision 0.9179	recall 0.9248	F1 0.9213
Contingency.Cause.Result                  precision 0.9841	recall 0.8378	F1 0.9051
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
EntRel                                    precision 0.7852	recall 0.9860	F1 0.8742
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.7500	F1 0.8571
Expansion.Conjunction                     precision 0.9597	recall 0.9286	F1 0.9439
Expansion.Instantiation                   precision 1.0000	recall 0.8596	F1 0.9245
Expansion.Restatement                     precision 0.9175	recall 0.8241	F1 0.8683
Temporal.Asynchronous.Precedence          precision 0.9718	recall 0.9324	F1 0.9517
Temporal.Asynchronous.Succession          precision 0.9773	recall 0.8431	F1 0.9053
Temporal.Synchrony                        precision 0.8485	recall 0.9655	F1 0.9032
Overall parser performance --------------
Precision 0.9162 Recall 0.9162 F1 0.9162

================================================
Evaluation for explicit discourse relations only
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.9514	recall 0.9514	F1 0.9514
Comparison.Concession                     precision 0.6667	recall 0.5000	F1 0.5714
Comparison.Contrast                       precision 0.9524	recall 0.9756	F1 0.9639
Contingency.Cause.Reason                  precision 0.9808	recall 0.9273	F1 0.9533
Contingency.Cause.Result                  precision 1.0000	recall 0.8421	F1 0.9143
Contingency.Condition                     precision 0.9783	recall 0.9574	F1 0.9677
Expansion.Alternative                     precision 0.8571	recall 1.0000	F1 0.9231
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Conjunction                     precision 0.9679	recall 0.9837	F1 0.9757
Expansion.Instantiation                   precision 1.0000	recall 1.0000	F1 1.0000
Expansion.Restatement                     precision 1.0000	recall 0.8333	F1 0.9091
Temporal.Asynchronous.Precedence          precision 0.9608	recall 1.0000	F1 0.9800
Temporal.Asynchronous.Succession          precision 1.0000	recall 0.8542	F1 0.9213
Temporal.Synchrony                        precision 0.8413	recall 0.9815	F1 0.9060
Overall parser performance --------------
Precision 0.9514 Recall 0.9514 F1 0.9514

================================================
Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Explicit connectives         : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 1 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg 2 extractor              : Precision 1.0000 Recall 1.0000 F1 1.0000
Arg1 Arg2 extractor combined : Precision 1.0000 Recall 1.0000 F1 1.0000
Sense classification--------------
*Micro-Average                            precision 0.8852	recall 0.8852	F1 0.8852
Comparison.Concession                     precision 1.0000	recall 0.6000	F1 0.7500
Comparison.Contrast                       precision 0.9740	recall 0.8523	F1 0.9091
Contingency.Cause.Reason                  precision 0.8780	recall 0.9231	F1 0.9000
Contingency.Cause.Result                  precision 0.9787	recall 0.8364	F1 0.9020
EntRel                                    precision 0.7852	recall 0.9860	F1 0.8742
Expansion.Alternative.Chosen alternative  precision 1.0000	recall 0.5000	F1 0.6667
Expansion.Conjunction                     precision 0.9459	recall 0.8468	F1 0.8936
Expansion.Instantiation                   precision 1.0000	recall 0.8333	F1 0.9091
Expansion.Restatement                     precision 0.9130	recall 0.8235	F1 0.8660
Temporal.Asynchronous.Precedence          precision 1.0000	recall 0.8000	F1 0.8889
Temporal.Asynchronous.Succession          precision 0.6667	recall 0.6667	F1 0.6667
Temporal.Synchrony                        precision 1.0000	recall 0.7500	F1 0.8571
Overall parser performance --------------
Precision 0.8852 Recall 0.8852 F1 0.8852
```

==================================================================

```
acdd91a9ce7bf8e1f990acd4b8cca1b2  dsenser/data/models/pdtb.sense.model.LSTMSenser
c852915080e526141beeec0fd02d9085  dsenser/data/models/pdtb.sense.model.SVDSenser
```
