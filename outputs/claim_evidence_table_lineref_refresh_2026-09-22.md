# Line-reference refresh for claim_evidence_table.md

Refreshed 22 September 2026 against commit 3b613c1 (branch backup/verification-session-2026-09-10). Report only: the only files written are this one and outputs/claim_evidence_table.md.

**Baseline.** The table was compiled on 10 September 2026 at about 18:20 (commits bec2f8c, d84ce77). Every chapter 1, 2, 3, 5, 6 and 7 reference lands on its quoted text at d84ce77, so that commit is the baseline for those chapters. Chapter 4's references do not fit any committed version: the table records an 860-line file that 'stops mid-sentence' at 'says which is which', which sits at line 858 of b2cb9f5 (10 Sep 10:39) and at 864 of bec2f8c, and the four 'never defined' labels exist beyond line 860 in every committed version. The compiled chapter 4 was therefore a truncated read of an uncommitted working copy between those two commits; b2cb9f5 is used as its baseline and its references carry a drift of one to three lines, absorbed by the text matching.

**Method.** For each reference the baseline lines were extracted and located in the current thesis_tex/chapterN.tex by exact line match, then fuzzy match (difflib ratio at or above 0.6 within a 500-line window), and every non-exact result was checked by hand with grep on the row's distinctive numbers, labels or phrases. Where a quoted sentence was rewritten the replacement sentence is given; where the content is gone it is marked REMOVED and the table keeps the old reference with a stale note. Two PART 2.5 citation lists and one PART 2.4 phrase wrap onto a second table line; those continuation tokens were refreshed by hand after the scripted pass.

## Inventory

| form | occurrences |
|:--|:--|
| F1 chapterN.tex:A-B | 54 |
| F2 `chapterN.tex`:A-B | 36 |
| F3 chN:A-B | 14 |
| F4 Line column | 181 |
| F5 bare in-cell mention | 23 |
| F6 PART 2.5 citation-site list | 37 |
| F7 unprefixed continuation token after a chapter ref (`chapter4.tex:302, 716`, `chapter5.tex:280-288, 398-408`, `` `chapter5.tex`:464, 472 ``) | 3 |
| total | 348 occurrences over 291 unique (chapter, range) keys |

Form F4 is the bare Line column of the PART 4 per-chapter tables; F5 is bare 'at NNN', 'line NNN' mentions inside cells and PART 7 items; F6 is the PART 2.5 list of citation sites. Forms with a tilde or an en-dash range: none found.

## Outcome

| confidence | unique keys | occurrences | action in the table |
|:--|:--|:--|:--|
| HIGH | 204 | 237 | replaced |
| MEDIUM | 63 | 81 | replaced |
| LOW | 1 | 1 | old reference kept, stale note appended |
| REMOVED | 24 | 29 | old reference kept, stale note appended |

## Mapping

Old and new are line numbers in the named chapter file. 'Table lines' are the lines of claim_evidence_table.md (before this refresh) carrying the reference.

| old ref | new lines | how located | confidence | table lines |
|:--|:--|:--|:--|:--|
| `chapter1.tex:121-124` | 141-144 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 572 |
| `chapter1.tex:125-126` | 145-146 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 573 |
| `chapter1.tex:126-128` | 146-148 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 574 |
| `chapter1.tex:143-150` | 163-175 | todo replaced by a sourced statement on the 1e12 lower edge (Pitts2019), grep on e12 and 'extension below' | MEDIUM | 575 |
| `chapter1.tex:554-559` | 648-653 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 576 |
| `chapter1.tex:563-572` | 677-683 | exact match; the conference-paper negative now at 682-683 | HIGH | 577 |
| `chapter1.tex:646-655` | 757-765 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 578 |
| `chapter1.tex:668-676` | 793-801 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 579 |
| `chapter2.tex:142-148` | 141-147 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 595 |
| `chapter2.tex:174-178` | 174-178 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 594 |
| `chapter2.tex:209-213` | 216-220 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 605 |
| `chapter2.tex:214-216` | 221-223 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 606 |
| `chapter2.tex:233-235` | 238-240 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 594 |
| `chapter2.tex:329-336` | 339-346 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 598 |
| `chapter2.tex:338-340` | 348-350 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 598 |
| `chapter2.tex:358-385` | 368-395 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 607 |
| `chapter2.tex:402-407` | 412-416 | grep 3115 and 3117 | HIGH | 600 |
| `chapter2.tex:422-430` | 436-444 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 601 |
| `chapter2.tex:436-439` | 459-461 | grep 'electron-driven route'; sentence rewritten | MEDIUM | 602 |
| `chapter2.tex:471-483` | 493-503 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 596 |
| `chapter2.tex:680-693` | 718-731 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 608 |
| `chapter2.tex:704-718` | 743-756 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 609 |
| `chapter2.tex:721-726` | 759-764 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 609 |
| `chapter2.tex:772-781` | 810-819 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 610 |
| `chapter2.tex:795-798` | 833-836 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 611 |
| `chapter2.tex:798-802` | 836-840 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 612 |
| `chapter2.tex:834` | 870 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 613 |
| `chapter2.tex:890` | 927 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 613, 979 |
| `chapter2.tex:921-924` | 958-960 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 614 |
| `chapter2.tex:944-956` | 1017-1029 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 603 |
| `chapter2.tex:957-959` | 1030-1032 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 604 |
| `chapter2.tex:996` | 1067 | grep 3.6e-11 in chapter 2 | HIGH | 870 |
| `chapter2.tex:1018-1032` | 1113-1140 | Anderson table relocated; numbers changed (4250 comparisons after the 2002 corrigendum) | MEDIUM | 615 |
| `chapter2.tex:1048-1052` | none (the n=6 140 a0 box sentence is no longer in chapter 2) | grep 140 a_0 and 'R-matrix box' return nothing | REMOVED | 616 |
| `chapter2.tex:1062-1065` | 1426-1429 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 599 |
| `chapter2.tex:1077-1079` | 1460-1462 | grep '201 of the 819' | HIGH | 597 |
| `chapter2.tex:1098-1128` | 1483-1512 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 617 |
| `chapter2.tex:1113-1115` | 1497-1499 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 626 |
| `chapter2.tex:1114` | 1498 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 886 |
| `chapter2.tex:1143-1164` | 1526-1546 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 618 |
| `chapter2.tex:1155-1164` | 1538-1546 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 905 |
| `chapter2.tex:1172-1174` | 1554-1556 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 619 |
| `chapter3.tex:218` | 222 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 638 |
| `chapter3.tex:222-225` | 226-235 | sentence rewritten: the three-fault claim is gone and 228-235 now state that only an A/gamma inconsistency moves the residual | MEDIUM | 141, 277, 639, 872, 930 |
| `chapter3.tex:267-272` | 278-283 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 114, 640 |
| `chapter3.tex:270` | 281 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 280 |
| `chapter3.tex:272` | 283 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 870 |
| `chapter3.tex:282-284` | 293-295 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 641 |
| `chapter3.tex:342-356` | 353-367 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 642 |
| `chapter3.tex:388-396` | 399-407 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 644 |
| `chapter3.tex:411-412` | 422-423 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 327, 646 |
| `chapter3.tex:441-454` | 464-479 | tau_relax and boxed M equations matched exactly at 472 and 477 | HIGH | 647 |
| `chapter3.tex:444-446` | 464-466 | sentence rewritten: v1 now described by PR = 2.64, 'two or three levels rather than across the manifold' | MEDIUM | 645, 954 |
| `chapter3.tex:493-497` | 518-521 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 648 |
| `chapter3.tex:507-509` | 532-534 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 649 |
| `chapter3.tex:519-526` | 544-549 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 650 |
| `chapter3.tex:519-543` | 544-567 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 430 |
| `chapter3.tex:528` | 552 | grep 'orders of'; now reads 'nine orders' | HIGH | 880, 952 |
| `chapter3.tex:528-529` | 552-554 | grep 'orders of'; now reads 'nine orders' | HIGH | 651 |
| `chapter3.tex:534-539` | 559-564 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 652 |
| `chapter3.tex:539-543` | 564-567 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 653 |
| `chapter3.tex:552-560` | 576-584 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 643 |
| `chapter3.tex:562-565` | 586-589 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 654 |
| `chapter3.tex:570` | none (the isolation-guard 'figure script' mention is gone) | grep 'figure script' finds one site only (1587) | REMOVED | 370 |
| `chapter3.tex:573-576` | 596-599 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 655 |
| `chapter3.tex:589-590` | 612-613 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 316, 656 |
| `chapter3.tex:605-608` | 628-631 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 657 |
| `chapter3.tex:630-633` | 715-718 | grep 'five orders'; passage moved into the non-normality section | MEDIUM | 658 |
| `chapter3.tex:633` | 717 | grep 'five orders' | HIGH | 879, 951 |
| `chapter3.tex:719-722` | 804-805 | exact match on the kappa range line | HIGH | 323, 660 |
| `chapter3.tex:809-812` | 894-897 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 662 |
| `chapter3.tex:809-822` | 894-907 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 306 |
| `chapter3.tex:814-822` | 899-907 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 466, 663 |
| `chapter3.tex:822-827` | 907-911 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 661 |
| `chapter3.tex:841-847` | 924-930 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 664 |
| `chapter3.tex:945-948` | 1040-1043 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 326 |
| `chapter3.tex:945-955` | 1040-1050 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 665 |
| `chapter3.tex:954` | 1046 | grep 'three decades'; the phrase is still present | MEDIUM | 881, 953 |
| `chapter3.tex:965-971` | 1060-1066 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 666 |
| `chapter3.tex:1017-1019` | 1112-1114 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 512, 540 |
| `chapter3.tex:1186-1187` | 1295-1296 | grep 'contain a negative entry' | HIGH | 325, 667 |
| `chapter3.tex:1191-1215` | 1302-1326 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 668 |
| `chapter3.tex:1255-1259` | 1373-1377 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 670 |
| `chapter3.tex:1268-1298` | 1386-1424 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 669 |
| `chapter3.tex:1285` | none (the second 'figure script' mention is gone) | grep 'figure script' finds one site only (1587) | REMOVED | 370 |
| `chapter3.tex:1320-1323` | 1571-1574 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 670 |
| `chapter3.tex:1338` | 1589 | exact match; the remaining 'figure script' mention is at 1587 | HIGH | 370 |
| `chapter3.tex:1377-1387` | 1636-1646 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 671 |
| `chapter3.tex:1396-1410` | 1650-1660 | grep Gate~D and SCD96; passage rewritten (see report: Gate D now diagnosed) | MEDIUM | 671 |
| `chapter3.tex:1535-1539` | 1789-1793 | exact match on 1790; 'nine orders' now reads 'nearly five orders' | MEDIUM | 659 |
| `chapter3.tex:1538` | 1791 | the 'orders of magnitude' line; now 'nearly five' | MEDIUM | 879, 951 |
| `chapter3.tex:1559-1560` | 1814-1816 | grep mu(L) > 0 'at every grid point' | HIGH | 324 |
| `chapter4.tex:8` | 8 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 692 |
| `chapter4.tex:161-164` | 174-180 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 696 |
| `chapter4.tex:164` | 177 | grep 3.61e-11; sentence expanded to 177-180 | HIGH | 870 |
| `chapter4.tex:168` | 184 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 697, 872 |
| `chapter4.tex:182-224` | 198-240 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 150, 698 |
| `chapter4.tex:192` | 208 | exact match on the baseline table row | HIGH | 696, 870 |
| `chapter4.tex:234-237` | none (todo closed, the fault-injection grid point is now recorded at 249-254) | no UNVERIFIED bracket left in chapter 4; the grid point is now recorded at 249-254 | REMOVED | 699, 870 |
| `chapter4.tex:266-275` | 294-302 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 700 |
| `chapter4.tex:280-285` | 308-312 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 701 |
| `chapter4.tex:300-309` | 328-338 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 702 |
| `chapter4.tex:300-313` | 328-343 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 349 |
| `chapter4.tex:302` | 330 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 362, 379 |
| `chapter4.tex:311-313` | 341-343 | grep 7.31e-9 and 2457 | MEDIUM | 703 |
| `chapter4.tex:314-316` | none (the 2457-vs-7371 comment is gone, resolved as 2457 comparisons at 342) | grep 7371 returns nothing; 342 now reads '2457 comparisons in total' | REMOVED | 703 |
| `chapter4.tex:327-334` | 356-363 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 704 |
| `chapter4.tex:348-364` | 377-398 | read in full: approach fraction 0.0123 to 0.0155, 1.5 percent, 'weak, and mis-named' all present | HIGH | 268, 705 |
| `chapter4.tex:379-390` | 412-425 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 706 |
| `chapter4.tex:384` | 416 | grep thesis_ready A1 | HIGH | 379 |
| `chapter4.tex:385` | 417 | grep 'anywhere is $86.8$' | HIGH | 533, 759, 871 |
| `chapter4.tex:408-412` | 444-448 | read in full | HIGH | 707 |
| `chapter4.tex:437-446` | 473-482 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 708 |
| `chapter4.tex:453-457` | 490-493 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 709 |
| `chapter4.tex:457` | 493 | grep findings_10 ADDENDUM B.1 | HIGH | 379 |
| `chapter4.tex:470-480` | 508-517 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 710 |
| `chapter4.tex:483-489` | 522-527 | grep verify_ch3_groupB; passage condensed | MEDIUM | 711 |
| `chapter4.tex:493-498` | 532-538 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 712 |
| `chapter4.tex:504-508` | 543-546 | grep 'refuses to draw' | HIGH | 713 |
| `chapter4.tex:517-522` | 553-557 | grep 1156 | HIGH | 714, 906 |
| `chapter4.tex:524-529` | 560-563 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 715, 907 |
| `chapter4.tex:531-536` | 566-570 | fuzzy match; sentence reordered | MEDIUM | 716 |
| `chapter4.tex:540-552` | 575-593 | exact match at both ends | HIGH | 717 |
| `chapter4.tex:548-550` | 585-591 | sentence rewritten: the 'likewise' justification is replaced by the theorem argument with |Delta| 1.945614 to 0.939493 | MEDIUM | 203, 278, 717, 933, 1006 |
| `chapter4.tex:550` | none (B.3 citation no longer in the tanh passage) | no findings_10 B.3 citation remains in the tanh passage (575-593); a B.3 cite sits at 481 in the superposition passage | REMOVED | 379 |
| `chapter4.tex:550-552` | 592-593 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 221 |
| `chapter4.tex:572-576` | 616-622 | fuzzy match 0.89 to 0.95; now names verify_operator_conditioning.py | HIGH | 323, 718 |
| `chapter4.tex:574` | 620 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 550 |
| `chapter4.tex:580` | none (bracket closed; kappa is now stamped by verify_operator_conditioning.py at 616-627) | no UNVERIFIED bracket left in chapter 4 | REMOVED | 877 |
| `chapter4.tex:580-587` | none (bracket closed; kappa is now stamped by verify_operator_conditioning.py at 616-627) | no UNVERIFIED bracket left in chapter 4 | REMOVED | 331, 719, 944 |
| `chapter4.tex:581` | none (C.1 citation gone with the closed bracket) | no findings_10 C.1 citation remains | REMOVED | 379 |
| `chapter4.tex:591-601` | 634-644 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 720 |
| `chapter4.tex:603-606` | 647-649 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 720 |
| `chapter4.tex:604` | 648 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 379 |
| `chapter4.tex:615-620` | 668-679 | grep 0.75 and ADDENDUM D.8; numbers changed sign (stamped scan) | MEDIUM | 721, 873 |
| `chapter4.tex:620` | 674 | grep findings_10 ADDENDUM D.8 | HIGH | 379 |
| `chapter4.tex:626-635` | 687-701 | read in full ('A trap in this test') | HIGH | 722 |
| `chapter4.tex:637-643` | 725-740 | grep 9.4, 10.7, 6.3; passage rewritten around the stamped run | MEDIUM | 723 |
| `chapter4.tex:688-701` | 779-809 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 724, 903 |
| `chapter4.tex:696` | none (no thesis_ready A6 citation remains; the 0.06 percent sentence is at 793) | grep thesis_ready finds A1 only (416, 438) | REMOVED | 379 |
| `chapter4.tex:711-717` | 826-844 | grep 0.263; the D.8 numbers are now quoted as an earlier draft's, replaced by a stamped run | MEDIUM | 351 |
| `chapter4.tex:716` | 838 | grep findings_10 ADDENDUM D.8 | HIGH | 379 |
| `chapter4.tex:742-798` | 869-928 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 725, 902 |
| `chapter4.tex:743` | 870 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 882 |
| `chapter4.tex:772` | 902 | grep findings_10 §3.1 (table caption) | HIGH | 379 |
| `chapter4.tex:826-847` | 959-988 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 726 |
| `chapter4.tex:830` | none (the §3.2 citation is gone from the lower-bound table caption) | no findings_10 §3.2 citation remains; the lower-bound caption is at 959-961 | REMOVED | 379 |
| `chapter4.tex:836` | 972 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 884 |
| `chapter5.tex:192-195` | 232-235 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 745 |
| `chapter5.tex:199-206` | 239-246 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 746 |
| `chapter5.tex:227-238` | 264-280 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 743, 898 |
| `chapter5.tex:231` | 268 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:240-245` | 282-287 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 744 |
| `chapter5.tex:244` | 286 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:262-266` | 316-320 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 747 |
| `chapter5.tex:280-288` | 334-341 | read in full; grid-edge statement at 337-339 | HIGH | 523, 748, 910 |
| `chapter5.tex:289-294` | 342-347 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 749 |
| `chapter5.tex:289-297` | 342-350 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 303 |
| `chapter5.tex:300-304` | 353-357 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 750 |
| `chapter5.tex:304` | 357 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:325-329` | 378-382 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 751 |
| `chapter5.tex:334-336` | 387-389 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 752 |
| `chapter5.tex:336-342` | 389-395 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 753 |
| `chapter5.tex:343` | 396 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:343-351` | 396-411 | grep 1.65e13; UNVERIFIED bracket replaced by the stamped verify_crest_subgrid.py result | MEDIUM | 754 |
| `chapter5.tex:346` | 398 | grep 1.65e13 | HIGH | 875 |
| `chapter5.tex:347` | none (bracket closed; site now cites a stamped run) | bracket gone; no findings_10 §B.3 citation at this site (399-411 cite validation/crest_subgrid/) | REMOVED | 381 |
| `chapter5.tex:354-357` | 414-417 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 755 |
| `chapter5.tex:357` | 417 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:389` | none (CH5_EVIDENCE §1 citation gone from the step-dependence passage) | grep CH5_EVIDENCE finds 769 only | REMOVED | 381 |
| `chapter5.tex:398-408` | 456-470 | union of the 398-406 and 406-408 mappings (step-dependence table and the sentence after it) | MEDIUM | 523 |
| `chapter5.tex:398-406` | 456-465 | grep 190.4; table gained columns, four-interval maximum now at [0,5] | MEDIUM | 756 |
| `chapter5.tex:406-408` | 468-470 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 757, 918 |
| `chapter5.tex:433-473` | 499-554 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 348 |
| `chapter5.tex:452-459` | 517-521 | exact match | HIGH | 758 |
| `chapter5.tex:461-473` | 529-554 | exact match | HIGH | 759 |
| `chapter5.tex:464` | 532 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 871 |
| `chapter5.tex:466-473` | 548-554 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 357, 530 |
| `chapter5.tex:472` | 552 | grep 'worst-case separation is $86.5$' | HIGH | 871 |
| `chapter5.tex:475-479` | none (bracket closed; producer now named at 556-560) | grep SOURCE REQUIRED returns nothing in chapter 5; 556-560 now name verify_2s_not_slow.py and validation/slow_subspace/ | REMOVED | 361, 760 |
| `chapter5.tex:507-510` | 590-593 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 761 |
| `chapter5.tex:510` | 593 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:518-525` | 608-611 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 762 |
| `chapter5.tex:592-610` | 748-761 | exact match on label and rows | HIGH | 763 |
| `chapter5.tex:616` | 769 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:617-623` | none (bracket closed; k columns now traced to verify_reservoir_gain.py at 770-778) | no UNVERIFIED bracket left in chapter 5; 770-778 now trace the k columns to verify_reservoir_gain.py | REMOVED | 764 |
| `chapter5.tex:627-640` | 845-857 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 765 |
| `chapter5.tex:628` | 845 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 381 |
| `chapter5.tex:653-659` | 869-875 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 766 |
| `chapter5.tex:659` | 875 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:662-669` | 878-885 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 762 |
| `chapter5.tex:686-699` | 911-939 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 767 |
| `chapter5.tex:698` | 938 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:705` | 945 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:716-721` | 950-957 | grep 97 percent; the 'no refinement can make that worse' sentence is withdrawn at 959-962 | MEDIUM | 768 |
| `chapter5.tex:720` | 954 | grep findings_10 §B.2 in the 97 percent passage | MEDIUM | 381 |
| `chapter5.tex:738-753` | 1054-1070 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 769 |
| `chapter5.tex:783-796` | 1304-1317 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 770 |
| `chapter5.tex:836-882` | 1362-1407 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 771, 900 |
| `chapter5.tex:844` | 1370 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:858` | 1379 | nearest findings_10 citation in the falsified-hypothesis passage | MEDIUM | 381 |
| `chapter5.tex:898-905` | 1431-1438 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 772, 899 |
| `chapter5.tex:911-914` | 1444-1447 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 773 |
| `chapter5.tex:916-922` | 1449-1455 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 774 |
| `chapter5.tex:924-931` | 1457-1464 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 775, 909 |
| `chapter5.tex:931` | 1464 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:947` | 1478 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 875 |
| `chapter5.tex:947-954` | 1478-1485 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 776 |
| `chapter5.tex:952` | 1483 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 381 |
| `chapter5.tex:969-975` | 1499-1504 | bracket gone; replaced by 'crest position to n_max is untested' at 1501 | MEDIUM | 721, 873 |
| `chapter5.tex:977-984` | 1512-1519 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 777 |
| `chapter5.tex:978` | 1507 | grep 'excludes the $54$ points' | MEDIUM | 874 |
| `chapter5.tex:982` | 1511 | grep findings_10 §B.3 | MEDIUM | 381 |
| `chapter5.tex:997-999` | 1532-1534 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 885 |
| `chapter5.tex:1079-1136` | 1614-1686 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 304 |
| `chapter5.tex:1152` | none ('seven points' is gone from chapter 5) | grep 'seven points' returns nothing in chapter 5 | REMOVED | 726, 884 |
| `chapter5.tex:1153` | none (citation site gone) | no findings_10 citation near the lower-bound passage (1755-1765) | REMOVED | 381 |
| `chapter5.tex:1176-1193` | 1788-1803 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 778 |
| `chapter5.tex:1195-1201` | 1849-1853 | sensitivities paragraph matched; the 201-vs-202 flag that preceded it is gone | MEDIUM | 878 |
| `chapter5.tex:1196` | none (citation site gone with the closed flag) | the ELM-count flag and its citation are gone | REMOVED | 381 |
| `chapter5.tex:1207` | 1853 | grep '$104$ points excluded' | MEDIUM | 777, 874 |
| `chapter5.tex:1222` | 1938 | grep findings_10 §3.1 | MEDIUM | 381 |
| `chapter5.tex:1234` | 1996 | grep 2332 | HIGH | 882 |
| `chapter5.tex:1243` | 2106 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 381 |
| `chapter5.tex:1255` | 2112 | grep 86.8 in the M-vs-eps section | HIGH | 871 |
| `chapter5.tex:1268-1291` | 2164-2183 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 779 |
| `chapter5.tex:1297-1302` | 2231-2234 | grep 'eight unnamed combinations'; bracket replaced by a sentence naming twelve bases | MEDIUM | 496 |
| `chapter5.tex:1298` | none (thesis_ready A11 citation gone; the eight-basis note is now at 2232-2234) | grep A11 finds only the file-header comment (line 11) | REMOVED | 381 |
| `chapter5.tex:1304-1308` | 2286-2289 | the Greenland citation the row refers to; the old range sat two lines past it and its own text is now at 2307-2309 | MEDIUM | 779 |
| `chapter5.tex:1333-1408` | 2337-2419 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 347 |
| `chapter5.tex:1367-1408` | 2374-2459 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 780 |
| `chapter5.tex:1403-1408` | none (bracket closed; subsection ends at 2453) | grep SOURCE REQUIRED returns nothing in chapter 5; the subsection now ends at 2453 | REMOVED | 780 |
| `chapter6.tex:142-144` | 139-141 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 798 |
| `chapter6.tex:142-159` | 139-153 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 945 |
| `chapter6.tex:146-159` | 144-153 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 799 |
| `chapter6.tex:161` | 158 | grep 'optical depth of'; value now 80 per cm | HIGH | 799 |
| `chapter6.tex:164-168` | 158-166 | passage rewritten: half-slab depth 201, escape 1.2e-3, and the 'independent reproduction' reading withdrawn | MEDIUM | 800 |
| `chapter6.tex:165` | 161 | escape-factor line, rewritten | MEDIUM | 799 |
| `chapter6.tex:169-170` | 167-170 | exact match on 167; now 'about 1.8 orders' | HIGH | 801 |
| `chapter6.tex:173-175` | 173-175 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 802 |
| `chapter6.tex:184-188` | 184-188 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 803 |
| `chapter6.tex:204-209` | 218-223 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 804 |
| `chapter6.tex:215-219` | 226-229 | grep 0.4162; gate list condensed to prose | MEDIUM | 805 |
| `chapter6.tex:220-225` | 229-231 | grep 1156 | MEDIUM | 806, 906 |
| `chapter6.tex:226-229` | 232-234 | grep 'trapping switched off' | MEDIUM | 807 |
| `chapter6.tex:236-237` | 239-240 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 804 |
| `chapter6.tex:241-297` | 246-307 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 808, 901 |
| `chapter6.tex:280-283` | 283-286 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 802 |
| `chapter6.tex:303` | 302 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 799 |
| `chapter6.tex:336-339` | 336-338 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 809 |
| `chapter6.tex:347-361` | 346-360 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 810 |
| `chapter6.tex:369-377` | 368-376 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 811 |
| `chapter6.tex:393-399` | 393-399 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 449, 812 |
| `chapter6.tex:448-474` | 454-483 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 813 |
| `chapter6.tex:514-518` | 639-643 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 814 |
| `chapter6.tex:541-554` | 664-677 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 815 |
| `chapter6.tex:565` | 688 | grep 'offers no'; sentence rewritten | HIGH | 876 |
| `chapter6.tex:565-567` | 688-690 | sentence rewritten: 'no direct calculation ... A sensitivity scenario can nonetheless be constructed' | MEDIUM | 816 |
| `chapter6.tex:620-639` | 794-809 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 817 |
| `chapter6.tex:629` | 801 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 887 |
| `chapter6.tex:641` | 816 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 817, 887 |
| `chapter6.tex:688-692` | 848-852 | grep Guillemaut, 5.2, 1.6 | HIGH | 818 |
| `chapter6.tex:707-711` | 868-870 | grep 'Fusion 62' | MEDIUM | 819 |
| `chapter6.tex:759` | 913 | grep 2300 | MEDIUM | 882 |
| `chapter6.tex:782-796` | 913-922 (closure table condensed to a sentence at 913-922; per-point values (1.6226 s, 15.4x) no longer in chapter 6) | grep 1.6226, 15.4, 0.015173 return nothing; only 41.4 survives at 916 | LOW | 820 |
| `chapter6.tex:822-825` | 935-942 | grep 0.23324; the [1,4] vs [0,4] label is now resolved there | MEDIUM | 820 |
| `chapter6.tex:855` | 975 | grep 8.3 | HIGH | 883 |
| `chapter6.tex:978` | 1233 | grep 2332 | HIGH | 882 |
| `chapter6.tex:1006-1009` | 1270-1272 | fuzzy match 0.60 to 0.75, all three numbers present | HIGH | 821 |
| `chapter6.tex:1039` | 1288 | grep 'population coefficient sits a factor'; 'factor-8' now reads 8.3 | MEDIUM | 883 |
| `chapter7.tex:46-49` | 46-49 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 839 |
| `chapter7.tex:66-68` | 66-68 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 840 |
| `chapter7.tex:93-99` | 94-98 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 841 |
| `chapter7.tex:112-116` | 113-116 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 842 |
| `chapter7.tex:142-146` | 149-153 | exact match; 6.3 percent now reads 6.74 percent | HIGH | 843, 904 |
| `chapter7.tex:149-152` | 154-158 | bracket gone; replaced by the provenance sentence naming verify_reservoir_gain.py and reservoir_gain_summary.csv | MEDIUM | 395 |
| `chapter7.tex:154-160` | 164-170 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 844 |
| `chapter7.tex:160-162` | 170-172 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 845 |
| `chapter7.tex:173-178` | 183-187 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 846 |
| `chapter7.tex:178-182` | 187-191 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | MEDIUM | 847 |
| `chapter7.tex:269-278` | 329-337 | grep 'not previously' and 'already in print'; paragraph rewritten | MEDIUM | 848 |
| `chapter7.tex:284-292` | none (bracket closed; Fujimoto Ch. 4 now read and discussed at 355-362) | grep SOURCE REQUIRED returns nothing in chapter 7; Fujimoto Ch. 4 is now discussed at 355-362 | REMOVED | 849 |
| `chapter7.tex:346-350` | 513-517 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 347 |
| `chapter7.tex:346-353` | 513-520 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact and fuzzy) | HIGH | 850 |
| `chapter7.tex:362` | 528-529 | grep 'factor of'; factor 3 to 5 now reads 6.1 (and 9.4) | MEDIUM | 876 |
| `chapter7.tex:362-366` | 525-533 | grep molecular_channel | MEDIUM | 816 |
| `chapter7.tex:362-371` | 525-540 | grep molecular_channel; bracket gone | MEDIUM | 851 |
| `chapter7.tex:363` | none (D.4 citation replaced by a stamped run at 531) | grep findings_10 returns nothing in chapter 7; 531 cites validation/molecular_channel/ | REMOVED | 383 |
| `chapter7.tex:367` | none (the D.4 sentence and the UNVERIFIED bracket are gone) | grep findings_10 returns nothing in chapter 7 | REMOVED | 383 |
| `chapter7.tex:400` | none ('factor-8' is gone from chapter 7) | grep 'factor-8' and 'factor of 8' return nothing in chapter 7 | REMOVED | 883 |
| `chapter7.tex:419-427` | 635-642 | line text matched against the 10 Sep baseline (d84ce77; b2cb9f5 for chapter 4) (exact) | HIGH | 852 |

## LOW and REMOVED items

- `chapter2.tex:1048-1052` (REMOVED): the n=6 140 a0 box sentence is no longer in chapter 2
- `chapter3.tex:570` (REMOVED): the isolation-guard 'figure script' mention is gone
- `chapter3.tex:1285` (REMOVED): the second 'figure script' mention is gone
- `chapter4.tex:234-237` (REMOVED): todo closed, the fault-injection grid point is now recorded at 249-254
- `chapter4.tex:314-316` (REMOVED): the 2457-vs-7371 comment is gone, resolved as 2457 comparisons at 342
- `chapter4.tex:550` (REMOVED): B.3 citation no longer in the tanh passage
- `chapter4.tex:580` (REMOVED): bracket closed; kappa is now stamped by verify_operator_conditioning.py at 616-627
- `chapter4.tex:580-587` (REMOVED): bracket closed; kappa is now stamped by verify_operator_conditioning.py at 616-627
- `chapter4.tex:581` (REMOVED): C.1 citation gone with the closed bracket
- `chapter4.tex:696` (REMOVED): no thesis_ready A6 citation remains; the 0.06 percent sentence is at 793
- `chapter4.tex:830` (REMOVED): the §3.2 citation is gone from the lower-bound table caption
- `chapter5.tex:347` (REMOVED): bracket closed; site now cites a stamped run
- `chapter5.tex:389` (REMOVED): CH5_EVIDENCE §1 citation gone from the step-dependence passage
- `chapter5.tex:475-479` (REMOVED): bracket closed; producer now named at 556-560
- `chapter5.tex:617-623` (REMOVED): bracket closed; k columns now traced to verify_reservoir_gain.py at 770-778
- `chapter5.tex:1152` (REMOVED): 'seven points' is gone from chapter 5
- `chapter5.tex:1153` (REMOVED): citation site gone
- `chapter5.tex:1196` (REMOVED): citation site gone with the closed flag
- `chapter5.tex:1298` (REMOVED): thesis_ready A11 citation gone; the eight-basis note is now at 2232-2234
- `chapter5.tex:1403-1408` (REMOVED): bracket closed; subsection ends at 2453
- `chapter6.tex:782-796` (LOW): closure table condensed to a sentence at 913-922; per-point values (1.6226 s, 15.4x) no longer in chapter 6
- `chapter7.tex:284-292` (REMOVED): bracket closed; Fujimoto Ch. 4 now read and discussed at 355-362
- `chapter7.tex:363` (REMOVED): D.4 citation replaced by a stamped run at 531
- `chapter7.tex:367` (REMOVED): the D.4 sentence and the UNVERIFIED bracket are gone
- `chapter7.tex:400` (REMOVED): 'factor-8' is gone from chapter 7

