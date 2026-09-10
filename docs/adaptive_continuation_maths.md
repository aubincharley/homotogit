# Adaptive continuation: the mathematics [AI-Generated]

Notation fixed once. $\theta$ weights; $\eta$ the continuation parameter;
$\sigma\in\mathbb R^{19}_{\ge0}$ the per-site Gaussian widths (the *row*);
$k$ the global optimizer-update index; $e$ the epoch index; $K$ the total
number of updates; $\alpha_k$ the learning rate; $L$ the empirical risk.

---

## 1. The homotopy

For a transformation family $T_\eta$ and network $f_\theta$,

$$H(\theta,\eta)\;=\;\frac1n\sum_{i=1}^n \ell\big(f_{\theta,\eta}(x_i),\,y_i\big),
\qquad \ell(v,y)=\log\!\Big(\sum_c e^{v_c}\Big)-v_y .$$

The contract is a distinguished endpoint $\eta^\star$ with

$$T_{\eta^\star}=\mathrm{Id}
\quad\Longrightarrow\quad
H(\theta,\eta^\star)=L(\theta)\ \ \text{exactly.}$$

Here $\eta=\sigma$ and $\eta^\star=0$. The deformation is applied to the
*activations*, so it deforms the **model**, not the data:
$f_{\theta,0}=f_\theta$.

At insertion site $\ell$ the width is scaled to the grid on which that site's
feature map lives,

$$\sigma_\ell(e)\;=\;q_\ell(e)\,g(e),
\qquad
q_\ell(e)=\frac{\text{spatial width at }\ell\text{ now}}
{\text{width at }\ell\text{ with an unreduced input}} ,$$

with $q_\ell\equiv1$ at constant resolution.

---

## 2. The operator and its natural coordinate

### 2.1 Discrete kernel

On a fixed support $|i|\le R$,

$$w_\sigma[i]=\frac{\exp\!\big(-i^2/(2\sigma^2)\big)}
{\sum_{j=-R}^{R}\exp\!\big(-j^2/(2\sigma^2)\big)},
\qquad
T_\sigma h=\big(w_\sigma\otimes w_\sigma\big)*h ,$$

applied depthwise (channels never mix), with $\sum_i w_\sigma[i]=1$ so constants
are preserved exactly and the operator is a convex combination, hence
range-preserving.

### 2.2 The coordinate $\phi$

Define the **tap ratio**

$$\boxed{\ \phi(\sigma)\;=\;\frac{w_\sigma[1]}{w_\sigma[0]}\;=\;\exp\!\Big(-\frac{1}{2\sigma^2}\Big)\ }
\tag{2.1}$$

$\phi$ measures how far the kernel is from a delta: $\phi\to0$ as
$\sigma\to0^+$ and $\phi\to1$ as $\sigma\to\infty$. It is the right coordinate
for a step rule because it is bounded and its scale does not depend on the grid.

Chain rule, from $\tfrac{d}{d\sigma}\big(-\tfrac12\sigma^{-2}\big)=\sigma^{-3}$:

$$\frac{d\phi}{d\sigma}=\frac{\phi}{\sigma^{3}}
\qquad\Longrightarrow\qquad
\frac{\partial L}{\partial\phi}
=\frac{\sigma^{3}}{\phi}\,\frac{\partial L}{\partial\sigma}.
\tag{2.2}$$

Note $\phi$ is **not** the heat time $a=\sigma^2/2$; the two give different
paths through the same set of operators.

### 2.3 Floor from finite precision

By $(2.1)$ the off-centre taps vanish in floating point once
$\phi(\sigma)<\epsilon$, i.e.

$$\phi(\sigma)<\epsilon
\iff
\sigma\;<\;\big(2\ln(1/\epsilon)\big)^{-1/2}.
\tag{2.3}$$

| threshold | $\epsilon$ | $\sigma$ bound |
|---|---|---|
| float32 rounding | $2^{-24}\approx5.96\times10^{-8}$ | $0.174$ |
| float32 min denormal | $\approx1.4\times10^{-45}$ | $0.0696$ |

Below the second bound the kernel **is** the identity and
$\partial L/\partial\sigma\equiv0$ identically: a shrinking $\sigma$ can never
escape, and no sensitivity signal exists. Hence a floor $\sigma_{\min}$ with an
explicit snap $\sigma\mapsto0$, and hence the informative band is bounded away
from zero.

---

## 3. The corrector

Gradient accumulation over microbatches of sizes $n_1,\dots,n_M$,
$\sum_m n_m=B$:

$$g_k=\sum_{m=1}^{M}\frac{n_m}{B}\,\nabla_\theta\widehat L_m(\theta_k)
=\nabla_\theta\Big(\tfrac1B\textstyle\sum_{i\in\text{batch}}\ell_i\Big),$$

exact including a short final microbatch. SGD with momentum $\rho$ and weight
decay $\lambda$:

$$v_{k+1}=\rho v_k+g_k+\lambda\theta_k,
\qquad
\theta_{k+1}=\theta_k-\alpha_k v_{k+1}.$$

Learning rate: linear warmup then cosine, a pure function of the **global**
index $k$,

$$\alpha_k=\begin{cases}
\alpha\,\dfrac{k+1}{K_w}, & k<K_w,\\[2mm]
\alpha_{\min}+(\alpha-\alpha_{\min})\,\dfrac{1+\cos(\pi p_k)}{2},
\quad p_k=\dfrac{k-K_w}{K-K_w}, & k\ge K_w .
\end{cases}
\tag{3.1}$$

**LR mass.** For $\alpha_{\min}=0$ and $K_w\ll K$,

$$\mathcal M \;=\;\sum_{k=0}^{K-1}\alpha_k\;\approx\;\frac{\alpha K}{2}.
\tag{3.2}$$

$\mathcal M$ is the natural budget scalar: it is invariant to trading epochs
against peak rate, and it is what decides whether the corrector can converge at
all.

---

## 4. Trigger candidates and their failure modes

A predictor–corrector continuation steps $\eta$ once the corrector has
converged at $\eta_k$. This requires an observable that plateaus.

### 4.1 Loss differences are confounded with the learning rate

To first order in one step,

$$L(\theta_{k+1})-L(\theta_k)\;\approx\;-\,\alpha_k\|g_k\|^2 .
\tag{4.1}$$

So a test $|\Delta L|<\varepsilon$ fires when $\alpha_k\|g_k\|^2<\varepsilon$ —
indistinguishably because $\|g_k\|$ is small (converged) or because $\alpha_k$
is small (schedule). Under $(3.1)$ $\alpha_k\to0$ by construction, so the test
eventually fires regardless of convergence. The remedy is to normalise,

$$\frac{L(\theta_{k-W})-L(\theta_k)}{\sum_{j=k-W}^{k}\alpha_j}\;\sim\;\|g\|^2 .$$

### 4.2 Gradient norm is confounded with the weight norm

For a layer whose output is batch-normalised, the loss is invariant to the scale
of that layer's weights: $L(cW)=L(W)$ for all $c>0$. Degree-zero homogeneity
gives two consequences — Euler's relation and the gradient scaling law:

$$\langle\nabla_W L,\,W\rangle=0,
\qquad
\nabla L(cW)=c^{-1}\nabla L(W).
\tag{4.2}$$

Hence $\|g\|\propto\|W\|^{-1}$. Weight decay drives $\|W\|$ down, which
**inflates** $\|g\|$ with no relation to convergence. The scale-invariant
combination is the product:

$$\big\|\nabla L(cW)\big\|\cdot\|cW\| \;=\;\big\|\nabla L(W)\big\|\cdot\|W\| .
\tag{4.3}$$

### 4.3 Other scale-free candidates

$$\cos_k=\frac{\langle g_{k-1},\,g_k\rangle}{\|g_{k-1}\|\,\|g_k\|},
\qquad
\varrho_k=\frac{\|\theta_k-\theta_{k-1}\|}{\|\theta_k\|}
=\frac{\alpha_k\|v_k\|}{\|\theta_k\|}.$$

$\cos_k$ is invariant under $(4.2)$ by construction. $\varrho_k$ carries
$\alpha_k$ explicitly and so inherits the confound of $(4.1)$.

### 4.4 The structural-minimum lemma

Let a detector hold a history buffer, reset on firing, and require
(i) $W+1$ observations before any comparison and (ii) $p$ consecutive
"stale" verdicts. Let observations be spaced $c$ updates apart, and let a
minimum dwell $D_{\min}$ apply.

> **Lemma.** If the stale predicate is satisfied at every comparison, the
> inter-fire spacing is exactly
> $$\Delta k=\max\big(D_{\min},\;c\,(W+p)\big).
> \tag{4.4}$$

*Proof.* After a reset, comparisons begin at observation $W+1$; $p$ consecutive
verdicts complete at observation $W+p$; that is $c(W+p)$ updates, subject to the
dwell floor. $\square$

The contrapositive is the diagnostic: **if the measured spacing equals $(4.4)$,
the predicate carried no information** and the hyperparameters, not the data,
are the schedule. Any candidate signal must be checked against $(4.4)$ before
its firing pattern is interpreted.

---

## 5. The transfer gap and the marginal-value rule

Abandon convergence. Ask instead whether *dwelling* still buys progress on the
target objective. With a fixed probe $P$,

$$\boxed{\ G(\theta)\;=\;L_P(\theta;\eta^\star)\;-\;L_P(\theta;\eta_k)\ }
\tag{5.1}$$

the loss at the target endpoint minus the loss under the configuration actually
being trained. Two structural facts:

$$G\big(\theta\big)\Big|_{\eta_k=\eta^\star}=0,
\qquad
\frac{dG}{dt}>0 \;\Longrightarrow\;
\text{training here is increasing the distance to the target.}$$

The first gives automatic termination; the second gives the rule. Writing
$G_j$ for the $j$-th measurement and

$$s_j\;=\;\frac{G_{j-W}-G_j}{\big|G_{j-W}\big|}
\tag{5.2}$$

for the *relative* shrink over a window of $W$ measurements — relative because
$G$ spans three orders of magnitude along the path — the trigger is

$$\boxed{\ \text{fire}\iff s_j<\varepsilon\ \text{ for } p \text{ consecutive } j\ }
\tag{5.3}$$

with $\varepsilon=0$ the pure sign test. This is a **marginal-value** criterion:
it is well posed even when nothing converges, which $(4.1)$–$(4.3)$ make
necessary.

$G$ conflates two effects — genuine function specialisation, and mismatch of
normalisation statistics accumulated at $\eta_k\ne\eta^\star$. Both vanish as
$\eta_k\to\eta^\star$ and both are real costs of dwelling at the wrong level,
but $(5.1)$ does not separate them.

---

## 6. The controller

State: row $\sigma\in\mathbb R^{19}$, stage-start index $k_0$, buffer
$\mathcal H$, stale counter $\mathrm{st}$, measurement counter $m$.

**Measure** every $c$ updates:

$$m\leftarrow m+1;\quad
\text{if } m\le B_{\text{out}}:\ \text{discard}
\quad\text{else}\quad \mathcal H\leftarrow \mathcal H\cup\{G\}, $$

the blackout $B_{\text{out}}$ discarding measurements immediately after a step,
where $G$ jumps discontinuously.

**Update the counter** by $(5.2)$:

$$\mathrm{st}\leftarrow
\begin{cases}
\mathrm{st}+1,& |\mathcal H|>W \text{ and } s<\varepsilon,\\
0,&\text{otherwise.}
\end{cases}$$

**Decide**, with $d=k-k_0$:

$$\text{advance}\iff
\underbrace{d\ge D_{\max}}_{\text{deadline}}
\ \ \vee\ \
\Big(\underbrace{d\ge D_{\min}}_{\text{dwell}}\wedge\ \mathrm{st}\ge p\Big).
\tag{6.1}$$

**On advance:** step the row (§7), then $k_0\leftarrow k$,
$\mathcal H\leftarrow\varnothing$, $\mathrm{st}\leftarrow0$, $m\leftarrow0$.

The guards are not decoration. $D_{\max}$ makes termination unconditional;
$D_{\min}$ bounds the fire rate; $B_{\text{out}}$ removes the post-step
transient; and holding $K$ fixed keeps $(3.1)$ — a function of $k$ alone —
identical across arms, so "adaptivity helped" is not confounded with "the
learning-rate path changed". Setting $D_{\min}=D_{\max}$ recovers a fixed
schedule exactly, which is the degenerate check.

---

## 7. The predictor step

Direction is **fixed by construction**, magnitude only is adaptive. The reason
is that

$$\frac{\partial L}{\partial\sigma_\ell}<0
\quad\text{occurs for a substantial fraction of sites,}$$

i.e. more smoothing would *lower* the loss — smoothing acts as a regulariser. A
descent rule on $\sigma$ therefore makes $\sigma$ a free variable in
$\min_{\theta,\sigma}L(\theta,\sigma)$, whose solution has no reason to lie at
$\sigma=0$; the exact endpoint, and with it the homotopy, would be forfeited.

Let $A=\{\ell:\sigma_\ell>0\}$, let $\gamma_\ell=|\partial L/\partial\phi_\ell|$
via $(2.2)$, and $\tilde\gamma=\operatorname{median}_{\ell\in A}\gamma_\ell$.
With $n=\max(\text{stages left},1)$:

$$r_\ell=\operatorname{clip}\!\Big(\frac{\tilde\gamma}{\gamma_\ell+\epsilon},
\ \kappa^{-1},\ \kappa\Big),
\tag{7.1}$$

$$\boxed{\;
d_\ell=\operatorname{clip}\!\Big(\frac{\sigma_\ell}{n}\,\delta_{\text{ref}}\,r_\ell,
\ d_{\min},\ d_{\max}\Big),
\qquad
\sigma_\ell\leftarrow\sigma_\ell-d_\ell \;}
\tag{7.2}$$

followed by the floor rule $\sigma_\ell<\sigma_{\min}\Rightarrow\sigma_\ell\mapsto
\min(\sigma_{\min},\sigma_\ell)$, and $\mapsto0$ once no stages remain.

Three readings of $(7.1)$–$(7.2)$:

**Normalisation is global, not per-site.** Dividing each site by its *own*
scale would send every ratio to $1$ and destroy the signal being measured;
dividing all sites by a common $\tilde\gamma$ fixes the units while preserving
relative structure. A site of median sensitivity takes the reference step.

**$\kappa$ bounds a positive feedback.** $r_\ell\propto\gamma_\ell^{-1}$, so an
insensitive site takes a larger step, becoming more insensitive still. The clip
to $[\kappa^{-1},\kappa]$ is the only thing bounding that loop.

**Absent signal reduces to a schedule.** Where no usable sensitivity exists —
which by §2.3 is *everywhere* once $\sigma<\sigma$-bound — the code takes
$r_\ell=1$ and $(7.2)$ becomes

$$\sigma_{j+1}=\sigma_j\Big(1-\frac{\delta_{\text{ref}}}{n}\Big),
\tag{7.3}$$

a **geometric** descent with ratio $1-\delta_{\text{ref}}/n$, clipped below by
$d_{\min}$. Two corollaries worth stating because they are easy to get wrong:

- $\delta_{\text{ref}}$ is **not** the step size; the step is
  $\sigma\delta_{\text{ref}}/n$. If $\sigma\delta_{\text{ref}}/n<d_{\min}$ every
  step clips to $d_{\min}$ and the walk becomes *linear* with slope $d_{\min}$,
  independent of $\delta_{\text{ref}}$.
- reaching $\sigma_{\min}$ from $\sigma_0$ under $(7.3)$ needs

$$J=\Big\lceil
\frac{\ln(\sigma_{\min}/\sigma_0)}{\ln\!\big(1-\delta_{\text{ref}}/n\big)}
\Big\rceil
\tag{7.4}$$

steps, which must be matched against the number of times $(6.1)$ actually
fires, otherwise the path is not traversed within budget.

### 7.1 Deadline ramp

Once the deadline binds, with $m$ stages remaining,

$$\sigma_\ell\leftarrow\sigma_\ell\Big(1-\frac1m\Big)
=\sigma_\ell\,\frac{m-1}{m},
\tag{7.5}$$

so $\sigma_\ell\to0$ **linearly** in exactly $m$ steps: $m,m-1,\dots,1$ gives
factors $\frac{m-1}{m}\cdot\frac{m-2}{m-1}\cdots\frac{0}{1}=0$. A run whose
deadline fires is partly a fixed schedule and must be reported as such.

---

## 8. Traversal budget

Let $F$ be the number of fires before the deadline and $\bar d$ the mean step.
The path is traversed iff

$$\sum_{j=1}^{F} \bar d \;\gtrsim\; \sigma_0-\sigma_{\min}
\qquad\Longleftrightarrow\qquad
F\,\bar d\;\gtrsim\;\sigma_0-\sigma_{\min}.
\tag{8.1}$$

$F$ is set by the trigger through $(6.1)$ and $\bar d$ by the stepper through
$(7.2)$. **They must be designed together**: a correct trigger with too small a
step under-traverses and hands the remainder to $(7.5)$; a correct step with too
fast a trigger exhausts the path immediately. Both failures produce a
trajectory dominated by something other than the intended rule.

---

## 9. Schedule scaling

A fixed piecewise-constant schedule with $\Lambda$ levels, dwell $\delta$ and
bypass at $E_\star$ over horizon $E$:

$$g(e)=\Lambda_{\min(\lfloor e/\delta\rfloor,\ \Lambda-1)}
\ \ (e<E_\star),\qquad g(e)=0\ \ (e\ge E_\star).$$

Scaling the horizon by $S$ preserves *shape* — not absolute epochs — under

$$\delta\mapsto S\delta,\qquad E_\star\mapsto S E_\star,\qquad E\mapsto SE,$$

which holds both the per-level dwell fraction $\delta/E$ and the terminal
fraction $1-E_\star/E$ invariant.

## 10. Convergence budget

By $(3.2)$, $\mathcal M\approx\alpha K/2$ with $K=E\cdot\lceil n/B\rceil$. For
$n=5\times10^4$, $B=128$ ($391$ updates per epoch):

| $E$ | $\alpha$ | $K$ | $\mathcal M$ | relative |
|---:|---:|---:|---:|---:|
| 30 | 0.005 | 11 730 | 29.3 | $1\times$ |
| 120 | 0.005 | 46 920 | 117.3 | $4\times$ |
| 120 | 0.05 | 46 920 | 1173 | $40\times$ |
| 40 | 0.1 | 14 040 | 702 | $24\times$ |

The last row is a configuration observed to reach training interpolation. So
quadrupling $E$ at fixed $\alpha$ leaves $\mathcal M$ a factor
$702/117.3\approx6$ short, whereas raising $\alpha$ tenfold at $E=120$ exceeds
it by $\approx1.7$. Convergence is governed by $\mathcal M$, not by $E$ alone —
which is why "more epochs" and "a convergence test" are not the same request.
