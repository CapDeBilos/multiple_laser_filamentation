# multiple_laser_filamentation
Electrodynamics project: Simulations of multiple laser filamentation for the PHY_2S004_EP electodynamics course.
Plasma Extension of the Propagation Model
Objective

We extend the paraxial envelope model by including plasma generation and its feedback on the electric field, allowing us to simulate laser filamentation in air.

Model Extension

Starting from the paraxial equation with diffraction, GVD, and Kerr nonlinearity, we add the plasma term:

−
𝜎
2
(
1
+
𝑖
𝜔
𝜏
)
 
𝜌
 
𝜀
−
2
σ
	​

(1+iωτ)ρε

which accounts for:

absorption of the field by free electrons
defocusing due to the plasma-induced refractive index change
Electron Density Dynamics

The electron density 
𝜌
(
𝑟
,
𝑡
)
ρ(r,t) is computed at each propagation step using:

∂
𝜌
∂
𝑡
=
𝜎
𝑛
𝑏
2
𝐸
𝑔
𝜌
∣
𝜀
∣
2
+
𝛽
(
𝐾
)
𝐾
ℏ
𝜔
∣
𝜀
∣
2
𝐾
−
𝛼
𝜌
2
∂t
∂ρ
	​

=
n
b
2
	​

E
g
	​

σ
	​

ρ∣ε∣
2
+
Kℏω
β
(K)
	​

∣ε∣
2K
−αρ
2

This includes:

avalanche ionization
multiphoton ionization
recombination
Numerical Implementation
The electric field envelope 
𝜀
(
𝑟
,
𝑡
)
ε(r,t) is propagated along 
𝑧
z using a Crank–Nicolson scheme for linear terms and an Adams–Bashforth scheme for nonlinear terms.
At each step:
Compute 
𝜌
(
𝑟
,
𝑡
)
ρ(r,t) from the current field
Evaluate plasma and Kerr nonlinear terms
Advance the field to the next 
𝑧
z-slice
The plasma equation is solved explicitly in time (Euler scheme).
Observations
The plasma term introduces a self-limiting mechanism:
Kerr effect increases intensity
plasma generation reduces it
This competition leads to:
intensity clamping
oscillatory propagation
onset of filamentation-like behavior
Numerical Considerations
The system is highly nonlinear and sensitive to parameters (especially 
𝛽
(
𝐾
)
β
(K)
)
Stability is ensured by:
small propagation steps
bounding the electron density
Outcome

This extension allows the model to capture the essential physics of laser filamentation in air, including the balance between nonlinear focusing and plasma defocusing.
