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

7. **Match the existing notebooks before writing a new one.** `examples/` has a fixed shape:
   title + `## Demonstrates` bullet list (with `Real-world:` items) + `References:` block;
   an imports cell; numbered `# N.` sections whose markdown opens with `Test H0: ... vs
   H1: ...`; code that seeds, names parameters with comments, builds a config, loops with
   `.update(...)`, breaks on `reject_null` with a `for/else`, then plots lettered `(a)-(d)`
   panels from `get_history_df()`; a type-I validity simulation; `Real World:` sections with
   **Scenario**, bullets, **Why sequential testing matters here** and **Realistic
   parameters**; a closing `# N. How It Works` with LaTeX. Read two of them first.

8. **Notebooks demonstrate the library, they do not extend it.** No new estimators,
   martingales, bounds or merging rules may be defined in a notebook cell - if a
   demonstration needs one, it belongs in `expectation/` with tests. Only data simulation
   (samplers, `rng.choice`, a buggy sampler being audited) and plotting are allowed inline.
   A "where to use this" table is not a substitute for a worked domain section.

9. **Verify every notebook number before writing the prose around it.** Prototype the
   experiments in a scratch script first: several plausible parameter sets produced no
   rejection at all (per-stream evidence below `log K` for the sparse merge) or contradicted
   the claim being made. Then state the result honestly, including the cases where the
   simpler method wins, and fix any README/AGENTS bullet that overclaims relative to it.

10. **Simulate the world, not the result.** Hardcoding what the method is supposed to
    discover - a fixed number of anomalies, a hand-written "degraded" probability vector,
    a covariance matrix typed in to be wrong - and then tuning horizons until the method
    looks good is cherry-picking, even when every number is individually plausible.
    Instead: draw the unknowns (anomaly indicators as Bernoulli, effect sizes from a
    distribution, change points uniformly), generate observables from a mechanism
    (ordered probit for ordinal scores, a common factor for correlated residuals, an
    actual biased sampler, injected corrupted records), fix horizons from domain
    reasoning, and report whatever comes out - including the runs that detect nothing.
    Setting the true parameters of a data-generating process (`theta_control = 0.10`) is
    fine and necessary; setting the answer is not.

11. **One seed is not a result.** Any claim of the form "this method detects X" needs
    replications with the detection rate and median stopping time, because a single run
    is one draw. Replicating overturned two conclusions here: the sparse-mixture merge
    alarms *later* than the arithmetic mean about half the time (its real gain is ~33
    nats of evidence, not latency), and a change-point failure is invisible to a Stein
    skeptic whose reference set was frozen on pre-change data. Reveal the ground truth
    only after reporting what the method said.

12. **Run every test against a null before trusting a detection.** Writing the handbook
    surfaced a real library defect: `SequentialTesting(test_type="variance")` rejects a
    TRUE null in 200/200 runs of pure N(0,1) noise, one- and two-sided (the mean path on
    identical data sits at 0.005). Two notebooks had already reported its "detections" as
    working features. A tool that always fires always looks insightful, so a null check is
    not optional diligence - it is the thing that distinguishes a result from an artefact.
    This is now a section in both notebooks rather than a silently corrected table.

13. **Tuning a horizon or an effect size until the method detects something is reward
    hacking, and it is easy to do by accident.** I extended an ad-tech flight from 30 to
    60 days and raised a fraud rate from 0.0008 to 0.0025 purely because the carefree list
    came back empty - each individually plausible, both chosen for the outcome. Fix every
    parameter from the business premise BEFORE running, run once, and report what comes
    out. Empty lists and undetectable shocks are findings: the 28-day flight retires
    nothing, the 10% marketplace shock is outside the null yet unreachable, and a 0.75
    visibility QRNG cannot be certified. Those are now the most instructive results in the
    handbook.

14. **"Realistic parameters" is not the same as "parameters I adjusted until it rejected."**
    Building the commercial handbook I caught myself running a *calibration sweep* over
    Stein correlation shifts and contamination drift levels, explicitly looking for the rung
    where the method detected something, then planning to write the section around that rung.
    Every individual value was defensible; the search procedure was not. The fix that also
    improved the notebook: stop picking a rung and **publish the whole ladder as an operating
    characteristic**, including the rungs with 0/50 detection. A power curve is more useful to
    a practitioner than a hand-picked success, and it cannot be reward-hacked. Corollary:
    write prose that *reads the computed result* (f-strings, printed tables) instead of
    asserting an outcome in markdown, so the text cannot drift from the run.

15. **A handbook teaches usage, not significance.** The user's framing: a reader arrives with
    a domain and needs to learn which capability owns their question and how to configure it.
    Whether the worked example rejects is irrelevant, and three of the ten cases here do not
    reject - a compliant payment portfolio that is nonetheless non-homogeneous across
    acquirers, a warranty book running *below* its reserving basis, and an LLM bake-off whose
    approved budget is ~9x too small to resolve the question. Those turned into the most
    instructive sections, because each one produces a real decision anyway.

16. **Check the null-generating code as carefully as the test.** My first "false escalation
    under a true global null" table reported 0.305 for a merge that is valid under arbitrary
    dependence - an impossible result that I nearly wrote up as a finding about the merge. The
    bug was mine: the shared macro factor shifted each vendor's residual mean, so the null was
    false by construction. Correct pattern for "dependent but null": couple the streams
    through a **mean-zero shared factor** (`rho*common + sqrt(1-rho^2)*idiosyncratic`), then
    verify the margins really are uniform (`P(p<0.05) ~ 0.05`) *and* that the dependence is
    real (`corr(p_i,p_j) > 0`) before drawing any conclusion. When a validity number looks
    wrong, suspect the simulation before the library.

17. **Escaping bites when generating notebooks programmatically.** Builder files used
    `A(code(r"""..."""))` so LaTeX survives in markdown, but inside a *raw* string `\"\"\"`
    stays literally backslashed and every code cell with a docstring became a SyntaxError -
    which surfaced as 38 cascading `NameError`s, not as an escaping problem. Two habits:
    unescape in the builder (`text.replace('\\"', '"')`), and **always ast.parse every code
    cell before executing the notebook** so the real error is reported once, at its source.

18. **Pandas attribute access silently shadows column names.** `df.product` resolves to
    `DataFrame.product`, not the column, and the failure appears far away as a matplotlib
    shape error. Use bracket access (`df["product"]`) for any column whose name could collide
    with a method - `product`, `mean`, `sum`, `min`, `max`, `corr`, `count`.

19. **Coverage is the deliverable in a handbook, and I shipped 34% of it.** The first version
    of `commercial-handbook.ipynb` used 52 of 154 public API items, carried three dead imports,
    passed string literals where the library ships typed enums, and hand-rolled charts that
    `utils.helper_functions` already draws. It read well and taught a third of the library. The
    user's framing is the right test: *"if you never use some of the capabilities, how would a
    reader know what to use them for?"* Before calling a demonstration notebook done, run the
    audit - AST-walk the package for public classes/functions, grep the notebook for each, and
    list what is missing. That one script turned a good notebook into a complete one (93%, or
    97% excluding the module that does not work).

20. **An unused capability is not neutral - it is a capability the reader will never adopt.**
    The fix is not a mention in a table. Each one needs the question it owns ("independence
    between groups but not within them" -> segment-product merge; "400k requests will not fit
    in a list" -> `OrderStatisticInterface`; "no reweighting of our scenarios explains this
    book" -> `PolytopeNull`). Write the commercial question first; the API call is the easy part.

21. **Batch fixes; re-runs are not free.** I spent four separate 13-minute notebook executions
    on one-line defects (a raw-string escaping artefact, a `df.product` method collision, a bad
    palette key, a batched-vs-single score callable) that a single cheap probe would have caught
    together. The user called this out twice. Rule: when a long verification loop fails,
    reproduce the failing call in a 5-second standalone script, fix *every* known issue, then
    re-run the expensive job once. Never re-run a 13-minute job to test a one-character change.
