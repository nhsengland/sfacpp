Formulae in ESASfaLcTre and ESASfaLcTreEM
1. Model structure
The LC-TRE model with $C$ classes mixes $C$ individual TRE models via a multinomial logit class-membership distribution.

Within class $c$, the TRE observation equation:

$$y_{it} = \mathbf{x}{it}'\boldsymbol{\beta}c + v{i0} + v{it} - s \cdot u_{it}$$

where $s = +1$ (production), $s = -1$ (cost), and:

$v_{it} \sim N(0, \sigma_{v,it}^2)$ — time-varying noise
$u_{it} \sim N^+(0, \sigma_{u,it}^2)$ — time-varying inefficiency (half-normal)
$v_{i0} \sim N(0, \sigma_{v,i0,t}^2)$ — firm-level random effect (time-varying variance allowed)
Heteroscedastic log-linear parameterisation (all three variances):

$$\sigma_{u,it}^2 = \exp(\mathbf{z}{uit}'\boldsymbol{\gamma}{u,c}), \quad \sigma_{v,it}^2 = \exp(\mathbf{z}{vit}'\boldsymbol{\gamma}{v,c}), \quad \sigma_{v,i0,t}^2 = \exp(\mathbf{z}{vi0,t}'\boldsymbol{\gamma}{v0,c})$$

2. Composite error and derived scalars
After integrating out $u_{it}$ conditionally on $v_{i0}$, the composite error at time $t$ given draw $v_{i0}^{(r)}$ is:

$$\varepsilon_{it}^{(r)} = y_{it} - \mathbf{x}_{it}'\boldsymbol{\beta}c - v{i0}^{(r)}$$

where the draw satisfies $v_{i0}^{(r)} = \sigma_{v,i0,t} \cdot \tilde{v}^{(r)}$ and $\tilde{v}^{(r)}$ is the standardised node (see §5).

Combined variance and ratio:

$$\sigma_{it}^2 = \sigma_{u,it}^2 + \sigma_{v,it}^2, \qquad \sigma_{it} = \sqrt{\sigma_{it}^2}, \qquad \lambda_{it} = \frac{\sigma_{u,it}}{\sigma_{v,it}}$$

3. Per-period conditional density (half-normal, panelLogLikHNormAt)
For each draw $\tilde{v}$, the period-$t$ log-density of the half-normal TRE model is:

$$\log f_{it}^{(r)} = \log 2 + \log \phi!\left(\frac{\varepsilon_{it}^{(r)}}{\sigma_{it}}\right) + \log \Phi!\left(\frac{-s,\varepsilon_{it}^{(r)},\lambda_{it}}{\sigma_{it}}\right) - \log \sigma_{it}$$

Panel log-likelihood for firm $i$, draw $r$:

$$S_i^{(r)} = \sum_{t=1}^{T_i} \log f_{it}^{(r)}$$

4. AGHQ posterior mode and spread (aghqModeAndSigma)
For each firm $i$ and class $c$, the unnormalised log-posterior over the standardised effect $v \sim N(0,1)$ is:

$$\log q(v) = S_i(v) - \frac{v^2}{2}$$

where $S_i(v) = \sum_t \log f_{it}$ evaluated with $\tilde{v} \equiv v$.

Newton's method (central finite differences, step $\delta = 10^{-3}$, max 10 iterations):

$$g = \frac{\log q(v + \delta) - \log q(v - \delta)}{2\delta}, \qquad h = \frac{\log q(v+\delta) - 2\log q(v) + \log q(v-\delta)}{\delta^2}$$

Newton update (clamped to $[-2, 2]$):

$$v \leftarrow v - \text{clamp}!\left(\frac{g}{h},,-2,,2\right)$$

Posterior precision and spread (floored):

$$\kappa = \max(-h,, 0.01), \qquad \sigma^* = \max!\left(\frac{1}{\sqrt{\kappa}},, 0.1\right)$$

5. AGHQ quadrature nodes (buildAGHQRow)
Standard GHQ is the change of variables $v = \sqrt{2}, x_k$ applied to $\int f(v),\phi(v),dv$. AGHQ adapts further by centering on the posterior mode $v^$ and scaling by $\sigma^$:

$$w_{ik}^{(k)} = v^* + \sigma^* \sqrt{2}, x_k, \qquad k = 1, \ldots, K$$

where $x_k$ are the raw Gauss-Hermite nodes (stored in rawNodes). The draws matrix for firm $i$ row is exactly ${w_{ik}^{(k)}}$.

6. AGHQ log-normalisation correction
The change of variable from the standard GHQ base to the adapted one introduces a Jacobian. The additive log-space correction relative to standard GHQ (which normalises by $\log\sqrt{\pi}$) is:

$$\Delta_i = \log \sigma^*_i$$

Full AGHQ normalised log-likelihood for firm $i$, class $c$:

The per-node corrected log-density is:

$$\ell_{ik} = S_i^{(k)} + z_k^2 - \frac{(w_{ik}^{(k)})^2}{2}$$

where $z_k^2$ is the standard GHQ exponent factor (from $e^{x_k^2}$ in the weight correction) and $-\tfrac{1}{2}(w_{ik}^*)^2$ is the N(0,1) prior evaluated at the adapted node. Then:

$$\log \hat{L}{ic} = \Delta_i + S{\max} + \log!\sum_k w_k \exp(\ell_{ik} - S_{\max}) - \log\sqrt{\pi}$$

where $S_{\max} = \max_k \ell_{ik}$ is the standard log-sum-exp stabiliser and $w_k$ are the GHQ weights.

7. Class membership probabilities (multinomial logit)
For $C$ classes, with covariates $\mathbf{z}_i$ and parameters $\boldsymbol{\alpha}_c$ ($c = 0, \ldots, C-2$; class $C-1$ is the reference):

$$\pi_{ic} = \frac{\exp(\mathbf{z}i'\boldsymbol{\alpha}c)}{\sum{c'=0}^{C-1} \exp(\mathbf{z}i'\boldsymbol{\alpha}{c'})}, \qquad \boldsymbol{\alpha}{C-1} \equiv \mathbf{0}$$

8. Observed log-likelihood (MSL path, operator())
Log-sum-exp over classes for firm $i$:

$$\log L_i = \log \sum_{c=0}^{C-1} \pi_{ic} \cdot \exp(\log \hat{L}_{ic})$$

Total observed log-likelihood:

$$\ell(\boldsymbol{\theta}) = \sum_{i=1}^{N} \log L_i$$

9. Posterior class weights (computePosteriors)
By Bayes' rule:

$$\tau_{ic} = \frac{\pi_{ic} \exp(\log \hat{L}{ic})}{\sum{c'} \pi_{ic'} \exp(\log \hat{L}_{ic'})}$$

Computed in log-space via logSumExp for numerical stability.

10. Gradient — segmentation parameters (gradHess, seg block)
Score contribution for firm $i$, class $c$ coefficient $\boldsymbol{\alpha}_c$ ($c < C-1$):

$$\frac{\partial \log L_i}{\partial \boldsymbol{\alpha}c} = (\tau{ic} - \pi_{ic}), \mathbf{z}_i$$

11. Gradient — class frontier parameters (gradHess, class block)
Score for firm $i$, class $c$ parameters $\boldsymbol{\theta}_c$:

$$\frac{\partial \log L_i}{\partial \boldsymbol{\theta}c} = \tau{ic} \cdot \nabla_{\boldsymbol{\theta}c} \log \hat{L}{ic}$$

The inner gradient $\nabla_{\boldsymbol{\theta}c} \log \hat{L}{ic}$ is computed by internalAnalyticJacHess. Its first-order terms are (where $\tilde{v}^{(r)} \equiv$ draw $r$, and $\mathcal{M}$ is the inverse Mills ratio):

$$A_{it}^{(r)} = -\frac{s,\varepsilon_{it}^{(r)}}{\sigma_{it}}\lambda_{it}, \qquad \mathcal{M}{it}^{(r)} = \frac{\phi(A{it}^{(r)})}{\Phi(A_{it}^{(r)})}, \qquad \partial_\beta^{(1)} = \frac{\varepsilon_{it}^{(r)}}{\sigma_{it}^2} + \frac{s,\mathcal{M}{it}^{(r)},\lambda{it}}{\sigma_{it}}$$

$$\frac{\partial \log f_{it}^{(r)}}{\partial \boldsymbol{\beta}c} = \partial\beta^{(1)} \cdot \mathbf{x}_{it}$$

$$\frac{\partial \log \sigma_{it}}{\partial \sigma_{it}} \text{ chain:}\quad f_\sigma^{(r)} = \frac{1}{\sigma_{it}}!\left[\left(\frac{\varepsilon_{it}^{(r)}}{\sigma_{it}}\right)^2 - \mathcal{M}{it}^{(r)} A{it}^{(r)} - 1\right], \qquad f_\lambda^{(r)} = -s,\mathcal{M}{it}^{(r)} \frac{\varepsilon{it}^{(r)}}{\sigma_{it}}$$

Then the log-variance chain rules through the heteroscedastic parameterisations:

$$\frac{\partial \log f_{it}^{(r)}}{\partial \boldsymbol{\gamma}{u,c}} = \left[f\lambda^{(r)} \cdot \tfrac{\lambda_{it}}{2} + f_\sigma^{(r)} \cdot \tfrac{\sigma_{u,it}^2}{2\sigma_{it}}\right]\mathbf{z}_{uit}$$

$$\frac{\partial \log f_{it}^{(r)}}{\partial \boldsymbol{\gamma}{v,c}} = \left[-f\lambda^{(r)} \cdot \tfrac{\lambda_{it}}{2} + f_\sigma^{(r)} \cdot \tfrac{\sigma_{v,it}^2}{2\sigma_{it}}\right]\mathbf{z}_{vit}$$

$$\frac{\partial \log f_{it}^{(r)}}{\partial \boldsymbol{\gamma}{v0,c}} = \partial\beta^{(1)} \cdot \tilde{v}^{(r)} \cdot \tfrac{\sigma_{v,i0,t}}{2} \cdot \mathbf{z}_{vi0,t}$$

12. BHHH (outer-product) Hessian approximation
$$\hat{H}(\boldsymbol{\theta}) = -\mathbf{J}^\top \mathbf{J}, \qquad \mathbf{J}_{i\cdot} = \frac{\partial \log L_i}{\partial \boldsymbol{\theta}}$$

13. Analytical Hessian — second-order scalars (from internalAnalyticJacHess)
The analytical second derivatives (accumulated as $\bar{H}i = \bar{h}i + \bar{h}i^{(2)}$) use the following intermediate second-order scalars. Let $D{it}^{(r)} = -A{it}^{(r)}\mathcal{M}{it}^{(r)} - (\mathcal{M}_{it}^{(r)})^2$:

$$\partial_{\beta\beta}^{(2)} = -\frac{1}{\sigma_{it}^2}!\left(1 - \lambda_{it}^2 D_{it}^{(r)}\right)$$

$$\partial_{\beta\sigma}^{(2)} = \frac{1}{\sigma_{it}^2}!\left(-2\frac{\varepsilon_{it}^{(r)}}{\sigma_{it}} - s\lambda_{it}(\mathcal{M}{it}^{(r)} + A{it}^{(r)} D_{it}^{(r)})\right)$$

$$\partial_{\beta\lambda}^{(2)} = \frac{s}{\sigma_{it}}!\left(\mathcal{M}{it}^{(r)} + A{it}^{(r)} D_{it}^{(r)}\right)$$

$$\partial_{\sigma\sigma}^{(2)} = -\frac{1}{\sigma_{it}^2}!\left(3!\left(\frac{\varepsilon_{it}^{(r)}}{\sigma_{it}}\right)^2 - A_{it}^{(r)}!\left(A_{it}^{(r)} D_{it}^{(r)} + 2\mathcal{M}_{it}^{(r)}\right) - 1\right)$$

$$\partial_{\lambda\lambda}^{(2)} = -\left(\frac{\varepsilon_{it}^{(r)}}{\sigma_{it}}\right)^2!\mathcal{M}{it}^{(r)}!\left(A{it}^{(r)} + \mathcal{M}_{it}^{(r)}\right)$$

$$\partial_{\sigma\lambda}^{(2)} = \frac{s,\varepsilon_{it}^{(r)}}{\sigma_{it}^2}!\left(\mathcal{M}{it}^{(r)} + A{it}^{(r)} D_{it}^{(r)}\right)$$

Cross-terms with $\sigma_{v,i0}$:

$$\partial_{\beta, v0}^{(2)} = \frac{\tilde{v}^{(r)}}{\sigma_{it}^2}!\left(\lambda_{it}^2 D_{it}^{(r)} - 1\right)$$

$$\partial_{\sigma, v0}^{(2)} = \frac{\tilde{v}^{(r)}}{\sigma_{it}^2}!\left(-2\frac{\varepsilon_{it}^{(r)}}{\sigma_{it}} - \lambda_{it}(\mathcal{M}{it}^{(r)} + A{it}^{(r)} D_{it}^{(r)})\right)$$

$$\partial_{\lambda, v0}^{(2)} = \frac{\tilde{v}^{(r)}}{\sigma_{it}}!\left(\mathcal{M}{it}^{(r)} + A{it}^{(r)} D_{it}^{(r)}\right)$$

$$\partial_{v0, v0}^{(2)} = -\left(\frac{\tilde{v}^{(r)}}{\sigma_{it}}\right)^2!\left(1 - \lambda_{it}^2 D_{it}^{(r)}\right)$$

Selected Hessian blocks (for parameters $\boldsymbol{\gamma}{v0,c}$, $\boldsymbol{\gamma}{u,c}$, $\boldsymbol{\gamma}_{v,c}$; each is an outer product scaled by a scalar):

$$H_{v0,v0} = \tfrac{1}{4}\sigma_{v,i0,t}!\left(\sigma_{v,i0,t}\cdot\partial_{v0,v0}^{(2)} + \partial_\beta^{(1)}\tilde{v}^{(r)}\right)\mathbf{z}{vi0}\mathbf{z}{vi0}'$$

$$H_{v0,,u} = \tfrac{\sigma_{v,i0,t}}{2}!\left(\partial_{\lambda,v0}^{(2)}\cdot\tfrac{\lambda}{2} + \partial_{\sigma,v0}^{(2)}\cdot\tfrac{\sigma_{u}^2}{2\sigma}\right)\mathbf{z}{vi0}\mathbf{z}{uit}'$$

$$H_{v0,,v} = \tfrac{\sigma_{v,i0,t}}{2}!\left(-\partial_{\lambda,v0}^{(2)}\cdot\tfrac{\lambda}{2} + \partial_{\sigma,v0}^{(2)}\cdot\tfrac{\sigma_{v}^2}{2\sigma}\right)\mathbf{z}{vi0}\mathbf{z}{vit}'$$

The full analytical Hessian per draw is accumulated as:

$$\bar{H}i = \sum_r Q{ir}\sum_t H_{it}^{(r)} + \sum_r Q_{ir}!\left(\mathbf{g}_{ir} - \bar{\mathbf{g}}i\right)!\left(\mathbf{g}{ir} - \bar{\mathbf{g}}_i\right)'$$

where $\bar{\mathbf{g}}i = \sum_r Q{ir},\mathbf{g}{ir}$ and $Q{ir}$ are the simulation weights (§14 below).

14. M-step weighted class objective (weightedClassLLAndGradHess)
The E-step provides firm-level posterior weights $\tau_{ic}$ and the AGHQ Q-weights $\tilde{Q}{ik}$ (posterior weights over quadrature nodes). The per-node corrected log-density using the frozen nodes $w{ik}^*$ is:

$$\ell_{ik} = S_i^{(k)} + [\text{node_corr}]_{ik} + \log w_k$$

$$[\text{node_corr}]{ik} = z_k^2 - \tfrac{1}{2}(w{ik}^*)^2$$

The within-firm M-step Q-weights (normalized posterior over nodes):

$$\tilde{Q}{ik} = \frac{\exp(\ell{ik} - S_{\max})}{\sum_k \exp(\ell_{ik} - S_{\max})}$$

The M-step class log-likelihood (complete-data expected log-likelihood):

$$\mathcal{Q}c(\boldsymbol{\theta}c) = \sum_i \tau{ic}!\left[\Delta_i + S{\max,i} + \log!\sum_k \exp(\ell_{ik} - S_{\max,i}) - \log\sqrt{\pi}\right]$$

Gradient:

$$\nabla_{\boldsymbol{\theta}c} \mathcal{Q}c = \sum_i \tau{ic} \sum_k \tilde{Q}{ik} \nabla_{\boldsymbol{\theta}c} \log f{it}^{(k)}$$

passed to internalAnalyticJacHess with $Q_{ir} = \tilde{Q}_{ik}$.

15. Segmentation M-step (mStepSeg)
Maximise $\mathcal{Q}(\boldsymbol{\alpha}) = \sum_i \sum_c \tau_{ic} \log \pi_{ic}(\boldsymbol{\alpha})$ via Newton–Raphson:

$$\nabla_{\boldsymbol{\alpha}c} \mathcal{Q} = \sum_i (\tau{ic} - \pi_{ic}),\mathbf{z}_i$$

$$\frac{\partial^2 \mathcal{Q}}{\partial \boldsymbol{\alpha}c \partial \boldsymbol{\alpha}d} = -\sum_i \pi{ic}(\delta{cd} - \pi_{id}),\mathbf{z}_i\mathbf{z}_i'$$

Update:

$$\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} - (H - 10^{-8} I)^{-1}\nabla\mathcal{Q}$$

16. EM convergence criterion
$$|\ell(\boldsymbol{\theta}^{(m+1)}) - \ell(\boldsymbol{\theta}^{(m)})| < \texttt{tol}$$

where $\ell$ is the observed log-likelihood evaluated via the AGHQ formula (§8 applied with GHQ quadrature).