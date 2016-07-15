Hyper-Parameters
================


Original Wang
=============

<!-- C=0.3, loss="hinge", penalty="l1", dual=True,
	multi_class="crammer_singer" -->

dev
---

Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6136 Recall 0.6136 F1 0.6136

Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8983 Recall 0.8983 F1 0.8983

Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.3632 Recall 0.3632 F1 0.3632

test
----

Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6078 Recall 0.6078 F1 0.6078

Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8946 Recall 0.8926 F1 0.8936

Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.3438 Recall 0.3445 F1 0.3442


Optimized Wang
==============

<!-- implicit: C=0.02, loss="hinge", penalty="l1", dual=True,
	multi_class="crammer_singer" -->
<!-- explicit: C=0.07, loss="hinge", penalty="l1", dual=True,
	multi_class="crammer_singer" -->

dev
---

Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6385 Recall 0.6385 F1 0.6385

Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.9165 Recall 0.9165 F1 0.9165

Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.3939 Recall 0.3939 F1 0.3939

test
----

Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6155 Recall 0.6155 F1 0.6155

Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8957 Recall 0.8937 F1 0.8947

Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.3567 Recall 0.3574 F1 0.3570


Original XGBoost
================

<!-- MAX_DEPTH = 3, NTREES = 300  -->

dev
---

Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6577 Recall 0.6577 F1 0.6577

Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.9196 Recall 0.9196 F1 0.9196

Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.4272 Recall 0.4272 F1 0.4272

test
----

Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6114 Recall 0.6114 F1 0.6114

Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8978 Recall 0.8959 F1 0.8969

Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.3468 Recall 0.3475 F1 0.3471

Optimized XGBoost
=================

<!-- MAX_DEPTH = 9, NTREES = 600  -->

dev
---

Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6520 Recall 0.6520 F1 0.6520

Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.9059 Recall 0.9059 F1 0.9059

Overall parser performance --------------
Precision 0.4286 Recall 0.4286 F1 0.4286

test
----

Evaluation for all discourse relations
Overall parser performance --------------
Precision 0.6262 Recall 0.6258 F1 0.6260

Evaluation for explicit discourse relations only
Overall parser performance --------------
Precision 0.8902 Recall 0.8883 F1 0.8893

Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)
Overall parser performance --------------
Precision 0.3817 Recall 0.3820 F1 0.3818
