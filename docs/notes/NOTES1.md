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
