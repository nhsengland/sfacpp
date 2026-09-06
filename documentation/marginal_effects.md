# Marginal Effects

Marginal effects are desirable to investigate the associated effect of a determinant or enviornmental variable on the inefficiency or efficiency. These determinants may enter through either the mean or variance, or both, of the inefficiency term.

## Stochastic frontier model with determinants

Under the assumption that the inefficiency term $u_i$ follows a truncated normal distribution, with pre-truncated mean $\mu_i$ and variance $\sigma_{ui}$. Under a half-normal distribution, $\mu_i = 0$. The stochastic noise component, $v_i$ follows a normal distribution with mean of zero and variance of $\sigma^2_{vi}$

$$y_i = \beta'x_i + v_i + u_i$$ (1)
$$u_i \sim N^+(\mu_i, \sigma^2_{ui}) $$ (2)
$$v_i \sim N(0, \sigma^2_{vi}) $$ (3)
$$ \mu_i = \delta_0 + \delta'z_i $$ (4)
$$ \sigma_{ui} = \exp(\gamma_0 + \gamma'z_i) $$ (5)
$$ \sigma_{vi} = \exp(\rho_0 + \rho'z_i) $$ (6)

Under Jondrow _et al_ (1982), the distribution of $u_i$ conditional on $\varepsilon_i = v_i-u_i$, is truncated normal, with the following mean and variance.

$$\tilde{\mu}_i = \frac{(\mu_i\sigma^2_{vi}-\varepsilon_i\sigma^2_{ui})}{\sigma^2_i}$$ (7)

$$\sigma^2_{*i} = \frac{\sigma^2_{ui}\sigma^2_{vi}}{\sigma^2_i}$$ (8)

Given that

$$\sigma^2_i=\sigma^2_{ui}+\sigma^2_{vi}$$ (9)

The JLMS point estimator of $u_i$ is 

$$E(u_i|\varepsilon_i)=\tilde{\mu}_i+\sigma_{*i}\frac{\phi(\tilde{\mu}_i/\sigma_{*i})}{\Phi(\tilde{\mu}_i/\sigma_{*i})}$$ (10)

and for point estimates of technical efficiency, this becomes as suggested by Battese and Coelli (1988),

$$E(\exp(-u_i|\varepsilon_i))=\exp(-\tilde{\mu}_i+\frac{\sigma^2_{*i}}{2})\frac{\Phi(\frac{\tilde{\mu}_i}{\sigma_{*i}}-\sigma_{*i})}{\Phi(\frac{\tilde{\mu}_i}{\sigma_{*i}})}$$ (11)

Where $\phi$ is the standard normal probability density function, and $\Phi$ is the standard normal cumulative density function.



## Wang (2002)

Wang (2002) 

## Olsen and Henningsen (2011)



## Kumbhakar and Sun (2013)

Kumbhakar and Sun (2013) derive the marginal effects $\frac{\partial E(u_i|\varepsilon_i)}{\partial z_{li}}$ based off the JLMS estimator in (10).

$$ m_i = \frac{\tilde{\mu}_i}{\sigma_{*i}} $$ (12)
$$ g_i = \frac{\phi(m_i)}{\Phi(m_i)} $$ (13)

Then the marginal effect of $l$th determinant on $E(u_i|\varepsilon_i)$ is

$$ \frac{\partial E(u_i|\varepsilon_i)}{\partial z_{li}} = \delta_l[\frac{\sigma^2_{vi}}{\sigma^2_i}(1-m_ig_i-g^2)] \\ + \gamma_l \frac{1}{\sigma^2_i}\{\sigma^2_{vi} \sigma_{*i} [g_i(1+m^2_i) + m_ig_i^2] - 2\sigma^2_{*i}(\varepsilon_i + \mu_i) (1-g^2_i-m_ig_i) \} \\ + \rho_l \frac{1}{\sigma^2_i} \{ \sigma^2_{ui} \sigma_{*i} [ g_i(1+m^2_i) m_ig^2_i ] + 2 \sigma^2_{*i}(\varepsilon_i + \mu_i)(1-g^2_i-m_ig_i) \}$$ (14)

## References

- Jondrow, ... (1982). _
- Kumbhakar, S., and Sun, K., (2013). _Derivation of marginal effects of determinants of technical inefficiency_.
- Olsen, J V., and Henningsen, A., (2011). _Investment utilitisation, adjustment costs, and technical efficiency in danish pig farms_.