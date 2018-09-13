# Knowledge Base Completion

## Data

### FreeBase

#### 1. FB15K-237

  - Download this dataset from [www.microsoft.com/en-us/download/details.aspx?id=52312](https://www.microsoft.com/en-us/download/details.aspx?id=52312)

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 272115 | 17535 | 20466 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 14541 | 14505 (99.75%) | 9809 (67.46%) | 10348 (71.16%) |  8 | 29 |

  - **Unseen_in_valid**: unseen entities in valid from a view of train

| #Relations  | #Relations_train | #Relations_valid | #Relations_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 237 | 237 (100.00%) | 223 (94.09%) | 224 (94.51%) | 0 | 0 |

  - **Unseen_in_valid**: unseen relations in valid from a view of train

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 6737 | 19332 | 50922 | 195124 |
| Valid | 237 | 1207 | 3749 | 12342 |
| Test | 302 | 1509 | 4291 | 14364 |
| All | 7276 | 22048 | 58962 | 221830 |

  - **n-to-n**: (head, reltaion, tail) is an n-to-n-type triple if there are more than one (head, realtion, \*) and also more than one (\*, relation, tail) in train+valid+test

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 35597 | 532 | 629 | 36758 |

  - **Interchangeable**: For each (head, relation, tail), the reversed (tail, relation, head) exists in train+valid+test

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 1625 | 90 | 91 | 1806 |

  - **Self-loop**: head == tail in a (head, relation, tail)

#### 2. FB15K-237c

(Clean version)

  - Remove unseen entities in valid and test
  - Remove self-loop triples

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 270490 | 17436 | 20347 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 14505 | 14505 (100.00%) | 9772 (67.37%) | 10287 (70.92%) |  0 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 234 | 234 (100.00%) | 218 (93.16%) | 220 (94.02%) | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 5130 | 19328 | 50918 | 195114 |
| Valid | 149 | 1205 | 3742 | 12340 |
| Test | 211 | 1505 | 4269 | 14362 |
| All | 5490 | 22038 | 58929 | 221816 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 33972 | 442 | 538 | 34952 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 |

#### 3. FB15K-237c-inv

(Adding inverse relations to FB15K-237c)

  - Add new triples by reversing existing triples, using a new relation name by the original's inverse `inv-{relation}` except for the interchangeable triples
  - Remove duplicated triples caused by adding reversed interchangeable triples in train or valid or test

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 507008 | 34674 | 40400 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 14505 | 14505 (100.00%) | 9772 (67.37%) | 10287 (70.92%) |  0 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 468 | 468 (100.00%) | 418 (89.32%) | 420 (89.74%) | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 10051 | 69599 | 69645 | 357713 |
| Valid | 290 | 4941 | 4940 | 24503 |
| Test | 418 | 5766 | 5764 | 28452 |
| All | 10759 | 80306 | 80349 | 410668 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 33972 | 686 | 782 | 35440 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 |

#### 4. FB15K-237c-fullKG

(FB15K-237c with full train as knowledge graph)

  - Make a 'graph.txt' by copying 'train.txt'

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 14505 | 270490 | 234 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 270490 | 17436 | 20347 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 14505 | 14505 (100.00%) | 9772 (67.37%) | 10287 (70.92%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 234 | 234 (100.00%) | 218 (93.16%) | 220 (94.02%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 5130 | 19328 | 50918 | 195114 |
| Valid | 149 | 1205 | 3742 | 12340 |
| Test | 211 | 1505 | 4269 | 14362 |
| All (excluding graph) | 5490 | 22038 | 58929 | 221816 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 33972 | 442 | 538 | 34952 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 |

#### 5. FB15K-237c-splitKG

(FB15K-237c with split train as knowledge graph (splitting train into graph:train with ratio 3:1))

  - Make a 'graph.txt' and a new 'train.txt' by splitting the original 'train.txt'

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 14381 | 202867 | 234 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 67623 | 17436 | 20347 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 14505 | 13467 (92.84%) | 9772 (67.37%) | 10287 (70.92%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 275 | 310 | 18 | 23 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 234 | 234 (100.00%) | 218 (93.16%) | 220 (94.02%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Graph | 3779 | 14525 | 38283 | 146280 |
| Train | 1351 | 4803 | 12635 | 48834 |
| Valid | 149 | 1205 | 3742 | 12340 |
| Test | 211 | 1505 | 4269 | 14362 |
| All | 5490 | 22038 | 58929 | 221816 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 25464 | 8508 | 442 | 538 | 34952 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 | 0 |

#### 6. FB15K-237c-inv-fullKG

(FB15K-237c-inv with full train as knowledge graph)

  - Make a 'graph.txt' by copying 'train.txt'

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 14505 | 507008 | 468 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 507008 | 34674 | 40400 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 14505 | 14505 (100.00%) | 9772 (67.37%) | 10287 (70.92%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 468 | 468 (100.00%) | 418 (89.32%) | 420 (89.74%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 10051 | 69599 | 69645 | 357713 |
| Valid | 290 | 4941 | 4940 | 24503 |
| Test | 418 | 5766 | 5764 | 28452 |
| All (excluding graph) | 10759 | 80306 | 80349 | 410668 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 33972 | 686 | 782 | 35440 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 |

#### 7. FB15K-237c-inv-splitKG

(FB15K-237c-inv with split train as knowledge graph (splitting train into graph:train with ratio 3:1))

  - Make a 'graph.txt' and a new 'train.txt' by splitting the original 'train.txt'

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 14381 | 380256 | 468 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 126752 | 34674 | 40400 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 14505 | 14046 (96.84%) | 9772 (67.37%) | 10287 (70.92%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 92 | 123 | 3 | 3 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 468 | 468 (100.00%) | 418 (89.32%) | 420 (89.74%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Graph | 7530 | 52255 | 52074 | 268397 |
| Train | 2521 | 17344 | 17571 | 89316 |
| Valid | 290 | 4941 | 4940 | 24503 |
| Test | 418 | 5766 | 5764 | 28452 |
| All | 10759 | 80306 | 80349 | 410668 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 25569 | 8403 | 686 | 782 | 35440 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 | 0 |

#### 8. FB15K-237c-inv-Disjoint-I

(Based on FB15K-237c-inv)

  - Combine train, valid and test into one
  - Split total triples into graph and `train/valid/test` with ratio 3:1
  - Split entities in `train/valid/test` into `train_entities`, `valid_entities` and `test_entities` with ratio 8:1:1
  - Make train, valid and test according to their entities (some triples might be lost due to cross-set entity-pairs)

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 14484 | 440110 | 468 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 85970 | 1898 | 1536 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 14493 | 11070 (76.38%) | 1004 (6.93%) | 881 (6.08%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 1004 | 881 | 1 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 468 | 464 (99.15%) | 234 (50.00%) | 235 (50.21%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 4 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Graph | 8735 | 63278 | 63412 | 304685 |
| Train | 1826 | 11545 | 11320 | 61279 |
| Valid | 23 | 379 | 413 | 1083 |
| Test | 32 | 214 | 218 | 1072 |
| All | 10616 | 75416 | 75363 | 368119 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 24794 | 5540 | 77 | 84 | 30495 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 | 0 |

#### 9. FB15K-237c-inv-Disjoint-II

(Based on FB15K-237c-inv)

  - Combine train, valid and test into one
  - Split total entities into `train/graph_entities`, `valid_entities` and `test_entities` with ratio 8:1:1
  - Make `train/graph`, valid and test according to their entities (some triples might be lost due to cross-set entity-pairs)
  - Split `train/graph` into train and graph with ratio 1:3

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 11480 | 282078 | 460 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 90994 | 6422 | 5166 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 13844 | 11105 (80.22%) | 1218 (8.80%) | 1125 (8.13%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 1218 | 1125 | 1218 | 1125 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 468 | 460 (98.29%) | 327 (69.87%) | 304 (64.96%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 6 | 2 | 6 | 2 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Graph | 6296 | 42223 | 42437 | 191122 |
| Train | 2084 | 13770 | 13557 | 61583 |
| Valid | 818 | 1937 | 1930 | 1737 |
| Test | 946 | 1452 | 1451 | 1317 |
| All | 10144 | 59382 | 59375 | 255759 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 17423 | 5685 | 396 | 266 | 23770 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 | 0 |

### WordNet

#### 1. WN18RR

  - Download this dataset from [github.com/TimDettmers/ConvE/blob/master/WN18RR.tar.gz](https://github.com/TimDettmers/ConvE/blob/master/WN18RR.tar.gz)
  
| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 86835 | 3034 | 3134 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 40943 | 40559 (99.06%) | 5173 (12.63%) | 5323 (13.00%) |  198 | 209 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 11 | 11 (100.00%) | 11 (100.00%) | 11 (100.00%) | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 9962 | 15819 | 40511 | 20543 |
| Valid | 365 | 581 | 1371 | 717 |
| Test | 350 | 547 | 1506 | 731 |
| All | 10677 | 16947 | 43388 | 21991 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 31821 | 1149 | 1153 | 34123 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 7 | 2 | 0 | 9 |

#### 2. WN18RRc

(Clean version)

  - Remove unseen entities in valid and test
  - Remove self-loop triples

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 86828 | 2822 | 2924 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 40559 | 40559 (100.00%) | 4845 (11.95%%) | 4987 (12.30%) | 0 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 11 | 11 (100.00%) | 11 (100.00%) | 11 (100.00%) | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 10015 | 15811 | 40469 | 20533 |
| Valid | 333 | 568 | 1206 | 715 |
| Test | 318 | 538 | 1338 | 730 |
| All | 10666 | 16917 | 43013 | 21978 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 31814 | 1145 | 1153 | 34112 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 |

#### 3. WN18RRc-inv

(Adding inverse relations to WN18RRc)

  - Add new triples by reversing existing triples, using a new relation name by the original's inverse `inv-{relation}` except for the interchangeable triples
  - Remove duplicated triples caused by adding reversed interchangeable triples in train or valid or test

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 143998 | 5610 | 5822 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 40559 | 40559 (100.00%) | 4845 (11.95%) | 4987 (12.30%) |  0 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test | #Unseen_in_valid | #Unseen_in_test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 19 | 19 (100.00%) | 19 (100.00%) | 19 (100.00%) | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 17119 | 50664 | 50669 | 25546 |
| Valid | 658 | 1768 | 1770 | 1414 |
| Test | 637 | 1871 | 1870 | 1444 |
| All | 18414 | 54303 | 54309 | 28404 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 33970 | 2256 | 2280 | 38506 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 |

#### 4. WN18RRc-fullKG

(WN18RRc with full train as knowledge graph)

  - Make a 'graph.txt' by copying 'train.txt'

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 40559 | 86828 | 11 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 86828 | 2822 | 2924 |

| #Entities | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 40559 | 40559 (100.00%) | 4845 (11.95%) | 4987 (12.30%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 11 | 11 (100.00%) | 11 (100.00%) | 11 (100.00%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 10015 | 15811 | 40469 | 20533 |
| Valid | 333 | 568 | 1206 | 715 |
| Test | 318 | 538 | 1338 | 730 |
| All (excluding graph) | 10666 | 16917 | 43013 | 21978 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 31814 | 1145 | 1153 | 34112 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 |

#### 5. WN18RRc-splitKG

(WN18RRc with split train as knowledge graph (splitting train into graph:train with ratio 3:1))

  - Make a 'graph.txt' and a new 'train.txt' by splitting the original 'train.txt'

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 38133 | 65121 | 11 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 21707 | 2822 | 2924 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 40559 | 23429 (57.77%) | 4845 (11.95%) | 4987 (12.30%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 1547 | 1664 | 222 | 218 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 11 | 11 (100.00%) | 11 (100.00%) | 11 (100.00%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Graph | 7544 | 11853 | 30335 | 15389 |
| Train | 2471 | 3958 | 10134 | 5144 |
| Valid | 333 | 568 | 1206 | 715 |
| Test | 318 | 538 | 1338 | 730 |
| All | 10666 | 16917 | 43013 | 21978 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 23762 | 8052 | 1145 | 1153 | 34112 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 | 0 |

#### 6. WN18RRc-inv-fullKG

(WN18RRc-inv with full train as knowledge graph)

  - Make a 'graph.txt' by copying 'train.txt'

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 40559 | 143998 | 19 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 143998 | 5610 | 5822 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 40559 | 40559 (100.00%) | 4845 (11.95%) | 4987 (12.30%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 19 | 19 (100.00%) | 19 (100.00%) | 19 (100.00%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Train | 17119 | 50664 | 50669 | 25546 |
| Valid | 658 | 1768 | 1770 | 1414 |
| Test | 637 | 1871 | 1870 | 1444 |
| All (excluding graph) | 18414 | 54303 | 54309 | 28404 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 33970 | 2256 | 2280 | 38506 |

| - | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 |

#### 7. WN18RRc-inv-splitKG

(WN18RRc-inv with split train as knowledge graph (splitting train into graph:train with ratio 3:1))

  - Make a 'graph.txt' and a new 'train.txt' by splitting the original 'train.txt'

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 40052 | 107998 | 19 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 36000 | 5610 | 5822 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 40559 | 29930 (73.79%) | 4845 (11.95%) | 4987 (12.30%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 1010 | 991 | 52 | 63 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 19 | 19 (100.00%) | 19 (100.00%) | 19 (100.00%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Graph | 12843 | 37976 | 38057 | 19122 |
| Train | 4276 | 12688 | 12612 | 6424 |
| Valid | 658 | 1768 | 1770 | 1414 |
| Test | 637 | 1871 | 1870 | 1444 |
| All | 18414 | 54303 | 54309 | 28404 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 25658 | 8312 | 2256 | 2280 | 38506 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 | 0 |

#### 8. WN18RRc-inv-Disjoint-I

(Based on WN18RRc-inv)

  - Combine train, valid and test into one
  - Split total triples into graph and `train/valid/test` with ratio 3:1
  - Split entities in `train/valid/test` into `train_entities`, `valid_entities` and `test_entities` with ratio 8:1:1
  - Make train, valid and test according to their entities (some triples might be lost due to cross-set entity-pairs)

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 40174 | 117520 | 19 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 24195 | 343 | 384 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 40429 | 21608 (53.45%) | 527 (1.30%) | 556 (1.38%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 527 | 556 | 1 | 3 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 19 | 19 (100.00%) | 14 (73.68%) | 17 (89.48%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Graph | 15085 | 41277 | 41178 | 19980 |
| Train | 3083 | 8418 | 8405 | 4289 |
| Valid | 50 | 126 | 114 | 53 |
| Test | 42 | 147 | 142 | 53 |
| All | 18260 | 49968 | 49839 | 24375 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 27039 | 6069 | 89 | 91 | 33288 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 | 0 |

#### 9. WN18RRc-inv-Disjoint-II

(Based on WN18RRc-inv)

  - Combine train, valid and test into one
  - Split total entities into `train/graph_entities`, `valid_entities` and `test_entities` with ratio 8:1:1
  - Make `train/graph`, valid and test according to their entities (some triples might be lost due to cross-set entity-pairs)
  - Split `train/graph` into train and graph with ratio 1:3

| #Nodes | #Edges | #Edges_types (#Relations) |
| :---: | :---: | :---: |
| 30389 | 75142 | 19 |

| #Triples_train | #Triples_valid | #Triples_test |
| :---: | :---: | :---: |
| 24240 | 1636 | 1540 |

| #Entities  | #Entities_train | #Entities_valid | #Entities_test |
| :---: | :---: | :---: | :---: |
| 33162 | 21460 (64.71%) | 1173 (3.54%) | 1094 (3.30%) |

| #Unseen_e_in_valid (from train's view) | #Unseen_e_in_test (from train's view) | #Unseen_e_in_valid (from graph's view) | #Unseen_e_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 1173 | 1094 | 1173 | 1094 |

| #Relations  | #Relations_train | #Relations_valid | #Relations_test |
| :---: | :---: | :---: | :---: |
| 19 | 19 (100.00%) | 18 (94.74%) | 18 (94.74%) |

| #Unseen_r_in_valid (from train's view) | #Unseen_r_in_test (from train's view) | #Unseen_r_in_valid (from graph's view) | #Unseen_r_in_test (from graph's view) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |

| Number of triples | #1-to-1 | #1-to-n | #n-to-1 | #n-to-n |
| :---: | :---: | :---: | :---: | :---: |
| Graph | 11125 | 26342 | 26340 | 11335 |
| Train | 3545 | 8516 | 8506 | 3673 |
| Valid | 910 | 356 | 356 | 14 |
| Test | 780 | 370 | 370 | 20 |
| All | 16360 | 35584 | 35572 | 15042 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Interchangeable_triples | 18281 | 5875 | 434 | 408 | 24998 |

| - | Graph | Train | Valid | Test | All |
| :---: | :---: | :---: | :---: | :---: | :---: |
| #Self-loop_triples | 0 | 0 | 0 | 0 | 0 |