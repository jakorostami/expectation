# Lessons (self-corrections from user feedback)

Session 2026-09-11, "three innovative ideas" task. Four rounds of pushback before the
scope was accepted; the patterns below are what each round corrected.

1. **Recognising a known object as an e-value is not a contribution.** "X is a likelihood
   ratio / supermartingale, therefore an e-process" (mSPRT, CTW ratio, Doob martingale of a
   fixed test, innovations, forecast scores) is trivially true. Only propose a construction if
   the foreign theorem *does work* and the result is an object the e-value literature lacks,
   with a computable certificate of validity. Apply this test before writing a proposal.

2. **Search the e-statistics literature before claiming novelty.** Two of three v4 proposals
   (sparse-anomaly mixture; sequential KSD) were published in 2025 by people in this community
   and I did not know it. Novelty claims must be preceded by a targeted search
   (arXiv + author pages of Ramdas, Grünwald, Vovk, Wang, Larsson/Ruf, Koning, Pérez-Ortiz,
   Shafer, Waudby-Smith). Then label honestly: "implementation of X (year) + extension Y".

3. **Cross-field ideas must land in the library's frame, not beside it.** v2's physics and
   coding stories were valid but disconnected; v3 over-corrected into narrative dressing.
   The right convergence is structural: express the construction as a strategy for Skeptic
   emitting sequential e-values, reuse the temporal betting layer and merging, and let the
   foreign content live inside the strategy. Protocol language is a means, not the product.

4. **Certificates over convergence.** When a solver produces the e-variable, divide by the
   exactly computed support function (or pay a violation tax) so validity does not depend on
   solver tolerance. Enumerate exactly whenever the object is finite; Monte Carlo only where
   it is not, with binomial tolerances stated.

5. **State novelty in three tiers and say which tier each item is in**: computational
   instantiation of existing theory; faithful implementation plus a documented extension;
   genuinely new object. Do not let framing blur the tier.

6. **Ask about paper retrieval only when it changes what can be claimed**; when the
   construction is derived and checked in-session, cite by author/year/venue without quoting
   unread equation numbers.
