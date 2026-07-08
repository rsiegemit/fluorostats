# Agreement statistics validation

Cross-checked `fluorostats.agreement` against independent numpy/scipy references on shared synthetic paired data (n=60).

**11/11 checks PASS**

| test | fluorostats | reference | abs_diff | PASS |
| --- | --- | --- | --- | --- |
| lins_ccc vs Lin(1989) closed form | 0.815708 | 0.815708 | 0 | True |
| lins_ccc vs r * C_b decomposition | 0.815708 | 0.815708 | 0 | True |
| icc vs two-way ANOVA ICC(A,1) | 0.818221 | 0.818221 | 2.22045e-16 | True |
| bland_altman bias vs mean(a-b) | -0.585133 | -0.585133 | 0 | True |
| bland_altman sd_diff vs std(a-b, ddof=1) | 1.01763 | 1.01763 | 0 | True |
| bland_altman loa_lower vs bias-1.96sd | -2.57969 | -2.57969 | 0 | True |
| bland_altman loa_upper vs bias+1.96sd | 1.40943 | 1.40943 | 0 | True |
| identical vectors: CCC == 1 | 1 | 1 | 0 | True |
| identical vectors: ICC == 1 | 1 | 1 | 0 | True |
| constant offset: Pearson r == 1 | 1 | 1 | 0 | True |
| constant offset: CCC < r (accuracy penalty) | 0.417227 | 1 | 0.582773 | True |
