## 🔹 Plasma Extension of the Propagation Model

### Objective

We extend the paraxial envelope model by including **plasma generation and its feedback on the electric field**, allowing us to simulate laser filamentation in air.

---

### Model Extension

Starting from the paraxial equation with diffraction, GVD, and Kerr nonlinearity, we add the plasma term:

$$
-\frac{\sigma}{2}(1+i\omega\tau)\,\rho\,\varepsilon
$$

which accounts for:

- **absorption** of the field by free electrons  
- **defocusing** due to the plasma-induced refractive index change  

---

### Electron Density Dynamics

The electron density $\rho(r,t)$ is computed at each propagation step using:

$$
\frac{\partial \rho}{\partial t}
=
\frac{\sigma}{n_b^2 E_g} \rho |\varepsilon|^2
+
\frac{\beta^{(K)}}{K \hbar \omega} |\varepsilon|^{2K}
-
\alpha \rho^2
$$

This includes:

- avalanche ionization  
- multiphoton ionization  
- recombination  

---

### Numerical Implementation

- The electric field envelope $\varepsilon(r,t)$ is propagated along $z$ using a **Crank–Nicolson scheme** for linear terms and an **Adams–Bashforth scheme** for nonlinear terms.
- At each step:
  1. Compute $\rho(r,t)$ from the current field
  2. Evaluate plasma and Kerr nonlinear terms
  3. Advance the field to the next $z$-slice

- The plasma equation is solved explicitly in time (Euler scheme).

---

### Observations

- The plasma term introduces a **self-limiting mechanism**:
  - Kerr effect increases intensity  
  - plasma generation reduces it  

This competition leads to:

- **intensity clamping**
- **oscillatory propagation**
- filamentation-like behavior  

---

### Numerical Considerations

- The system is highly nonlinear and sensitive to parameters (especially $\beta^{(K)}$)
- Stability is ensured by:
  - small propagation steps
  - bounding the electron density

---

### Outcome

This extension allows the model to capture the essential physics of **laser filamentation in air**, including the balance between nonlinear focusing and plasma defocusing.
