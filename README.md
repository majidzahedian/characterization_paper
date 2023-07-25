# telegraph_bridges

**Title: Accurate determination of post-selected based initialisazion fidelity.**

- Motivation
  For initialization by measurements this is important in
  Qubits - Gambetta,MArtinis  
  
  https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.112.190504
  
  etc., NV centers, low temperature SSR, charge state, nuclear spin fidelity. 
  The known solution is to determine fidelity of the SSR, and multiply it by the exp(-tau_ssr*gamma) which is only an approximation. 
  
- Other methods 
  - Machine Learning https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.200501
  - Adaptive schemes https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.120502

- Interesting paper on the formula from known way to calculate the telegraph process occupation times (what was done before):

  - Occupation of time distribution of telegraph processes)[https://www.sciencedirect.com/science/article/pii/S0304414911000755]
  - Occupation time distribution of Levy bridges: https://www.sciencedirect.com/science/article/pii/030441499500013W

- It was done before to estimate the SSR fidelity for charge state, and reliably estimate the rates of papers from Lukin, Hopper, Capellaro[https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.063408]. 
- Gambetta - filtration https://arxiv.org/abs/cond-mat/0701078

- Here we investigate in similar settings the way to optimize the fidelity of the postselection, and its accurate value. 
  - We look at the process from point of view of measurement output conditional on the final state after the measurement is done. We employ the formalism of telegraph processes pined to the final state, a subclass of brownian bridge process. 
  

