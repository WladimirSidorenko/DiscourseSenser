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