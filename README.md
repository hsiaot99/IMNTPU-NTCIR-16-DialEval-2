# NTCIR-16-DialEval-2
For details of the NTCIR-16 Dialogue Evaluation Task (DialEval-2), see [here](https://dialeval-2.github.io/DCH-2/taskdetails).

Recently, many reserachers are trying to build automatic helpdesk systems. However, there are very few methods to evaluate such systems. In DialEval-2, we aim to explore methods to evaluate task-oriented, multi-round, textual dialogue systems automatically. This dataset have the following features:
* Chinese customer-helpdesk dialogues carwled from Weibo.
* English dialgoues: manually translated from the Chinese dialgoues.
* Nugget type annotatoins for each turn: indicate whether the current turn is useful to accomplish the task.
* Quality annotation for each dialogue.
  * task accomplishment
  * customer satisfaction
  * dialogue effectiveness

In DialEval-2, we consider annotations ground truth, and participants are required to predict nugget type for each turn (Nugget Detection, or ND) and dialogue quality for each dialogue (Dialogue Quality, or DQ).
