####

As per Greene (2012, pp671), the same Halton draws are used for the log-likelihood function, or derivatives, or any function involving them, for observation `i`.


```math
% overall log likelihood function
\frac{\partial\text{ln}L_S}{\partial\bm{\theta}}=
\sum^n_{i=1}\frac{ 
    \frac{1}{R} \sum^R_{r=1} \frac{\partial( \prod^{T_i}_{t=1} P_{itr}(\bm{\theta}))}{\partial\bm{\theta}} 
}{
    \frac{1}{R} \sum^R_{r=1} \prod^{T_i}_{t=1} P_{itr}(\bm{\theta}) 
}
\\[3ex]
% simplification (15-20)
\frac{ \partial \prod^{T_i}_{t=1} P_{itr}(\bm{\theta}) }{\partial \bm{\theta}} = 
P_{ir}(\bm{\theta}) \sum^{T_i}_{t=1}\bm{g}_{itr}(\bm{\theta}) = 
P_{ir}(\bm{\theta})\bm{g}_{ir}(\bm{\theta})
\\[3ex]
% 15-21
\\[3ex]
\text{which simplifies to}
\\[3ex]
\frac{\partial\text{ln}L_S}{\partial\bm{\theta}} =\sum^{n}_{i=1}
\frac{
    \sum^{R}_{r=1} P_{ir}(\bm{\theta})\bm{g}_{ir}(\bm{\theta})
}{
    \sum^{R}_{r=1} P_{ir}(\bm{\theta})
}
% define weight Q_ir
\\[3ex]
\text{defining weight } Q_{ir}(\bm{\theta})
\\[3ex]
Q_{ir}(\bm{\theta})=\frac{P_{ir}(\bm{\theta})}{\sum^{R}_{r=1}P_{ir}(\bm{\theta})}
\\[3ex]
\text{such that}
\\[3ex]
0 < Q_{ir}(\bm{\theta}) < 1 \text{ and } \sum^{R}_{r=1}Q_{ir}(\bm{\theta}) = 1
```


```math

\text{For the second derivatives (hessian), given by } \bm{H}_{itr}(\bm{\theta}) = \frac{\partial^2 \text{ln} P_{itr}(\bm{\theta})}{\partial\bm{\theta}\partial \bm{\theta^{'}}} \text{ let }
\\[3ex]
\bm{H}_{ir}(\bm{\theta}) = \sum^{T_i}_{t=1} \bm{H}_{itr}(\bm{\theta})
\text{ and}
\\[3ex]
\bar{\bm{H}}_i(\bm{\theta}) = \sum^{R}_{r=1}Q_{ir}(\bm{\theta})\bm{H}_{ir}(\bm{\theta})
\\[3ex]
\text{Greene (2012, pp672) show the second derivatives matrix as three components}
\\[3ex]
\frac{\partial^2\text{ln} L_S}{\partial \bm{\theta} \partial \bm{\theta^{'}}} = \sum^{n}_{i=1} \lbrack 

\frac{ \sum^{R}_{r=1} P_{ir}(\bm{\theta}) \bm{H}_{ir} (\bm{\theta}) }{ \sum^{R}_{r=1} P_{ir} (\bm{\theta})} +
\frac{ 
    \sum^{R}_{r=1} P_{ir}(\bm{\theta}) \bm{g}_{ir} (\bm{\theta}) \bm{g}_{ir}(\bm{\theta})\bm{^{'}} 
}{ \sum^{R}_{r=1} P_{ir}(\bm{\theta}) } -
\frac{
    \lbrack \sum^{R}_{r=1} P_{ir}(\bm{\theta}) \bm{g}_{ir}(\bm{\theta}) \rbrack
    \lbrack \sum^{R}_{r=1} P_{ir}(\bm{\theta}) \bm{g}_{ir}(\bm{\theta}) \rbrack ^{'}
 }{ 
    \lbrack \sum^{R}_{r=1} P_{ir}(\bm{\theta}) \rbrack ^2
 }
\rbrack
\\[3ex] 
\text{Which can subsequently be rearranged as}
\\[3ex]
\frac{\partial^2\text{ln} L_S}{\partial \bm{\theta} \partial \bm{\theta^{'}}} = \sum^{n}_{i=1} \lbrace
\bar{\bm{H}}_{i}(\bm{\theta}) + \sum^{R}_{r=1}Q_{ir}(\bm{\theta})
\lbrack
\bm{g}_{ir}(\bm{\theta})-\bar{\bm{g}}_{i}(\bm{\theta})
\rbrack
\lbrack
\bm{g}_{ir}(\bm{\theta})-\bar{\bm{g}}_{i}(\bm{\theta})
\rbrack ^{'}
\rbrace
```