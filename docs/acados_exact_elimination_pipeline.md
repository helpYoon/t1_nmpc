# Exact Elimination Pipeline for Contact Acceleration Constraints in acados

## Context

You verified that `ZeroAccel` is affine in generalized acceleration. That is important, because it means you do **not** need to keep `ZeroAccel` as a hard nonlinear path equality in acados.

Instead of giving acados a model like

\[
\dot v = a, \qquad h_{\text{ZeroAccel}}(q,v,a)=0,
\]

with `ZeroAccel` enforced through

\[
\ell_h = u_h = 0,
\]

you can construct the model so that the generalized acceleration is **already contact-consistent**:

\[
a = a(q,v,u,p),
\]

and therefore

\[
h_{\text{ZeroAccel}}(q,v,a(q,v,u,p)) \equiv 0.
\]

Then `ZeroAccel` disappears from `model.con_h_expr`, so it never enters HPIPM as a hard path equality.

This is the exact-elimination analogue of what you wanted from OCS2-style projection, but performed at the nonlinear model level before acados forms the QP.

---

## 1. Your weighted acceleration projection understanding

You wrote the following formulation:

\[
a
=
\arg\min_a
\frac12(a-w_a)^\top W(a-w_a)
\quad
\text{s.t.}
\quad
\Sigma A a = -\Sigma b.
\]

This is correct as a **weighted projection of a desired acceleration** \(w_a\) onto the masked contact-acceleration manifold.

Define

\[
\bar A = \Sigma A,
\qquad
\bar b = \Sigma b,
\]

and

\[
G = \bar A W^{-1}\bar A^\top + \operatorname{diag}(1-s),
\]

where \(s\in\{0,1\}^m\) is the row-activation mask.

Then the KKT elimination gives

\[
a_0 = -W^{-1}\bar A^\top G^{-1}\bar b,
\]

\[
P = I-W^{-1}\bar A^\top G^{-1}\bar A,
\]

and therefore

\[
a = a_0 + P w_a.
\]

Here:

- \(P\) is the projector onto \(\ker(\bar A)\) in the \(W\)-metric.
- \(a_0\) is a particular solution satisfying the active constraint right-hand side.
- The inactive rows are zeroed by \(\Sigma\), and \(\operatorname{diag}(1-s)\) keeps \(G\) invertible for those inactive rows.
- For inactive rows, the corresponding multiplier is forced to zero.

For active rows, the reconstructed acceleration satisfies

\[
\bar A a + \bar b = 0.
\]

This is mathematically correct.

---

## 2. The major caveat: acceleration projection alone is kinematic

The acceleration projection

\[
\dot v = a_0(q,v)+P(q)w_a
\]

enforces contact acceleration consistency, but by itself it does **not** enforce rigid-body dynamics:

\[
M(q)\dot v+h(q,v)=S^\top\tau+J_c(q)^\top\lambda.
\]

With a constant \(W\), there is no mass matrix, Coriolis term, gravity, torque, or physical contact wrench in the model. The optimizer can choose base accelerations that satisfy the kinematic contact task but are not physically realizable.

So the pure acceleration projection is useful as a prototype or kinematic MPC model, but it is not a faithful whole-body balance model unless the rigid-body dynamics are included elsewhere.

The clean dynamic solutions are:

1. **Torque-input constrained forward dynamics**.
2. **Inverse-dynamics elimination in variables \(y=[a;f_c]\)**.

---

## 3. Pipeline A: torque-input constrained forward dynamics

Use this if you want the MPC input to be actuator torque:

\[
u = \tau.
\]

The state is

\[
x =
\begin{bmatrix}
q\\
v
\end{bmatrix}.
\]

The dynamics are obtained from the constrained equations of motion.

### 3.1 Active contact acceleration equation

For stance contacts,

\[
J_c(q)a+\dot J_c(q,v)v=0.
\]

With Baumgarte or PD stabilization,

\[
J_c(q)a
+
\dot J_c(q,v)v
+
K_d e_v(q,v)
+
K_p e_x(q)
=0.
\]

Write this as

\[
A(q)a+b(q,v)=0,
\]

where

\[
A(q)=J_c(q),
\]

and

\[
b(q,v)=\dot J_c(q,v)v+K_d e_v(q,v)+K_p e_x(q).
\]

### 3.2 Masked fixed-size contact system

Let \(A_{\text{all}}\) stack all possible contact rows, for example left and right foot rows:

\[
A_{\text{all}}(q)=
\begin{bmatrix}
A_L(q)\\
A_R(q)
\end{bmatrix},
\qquad
b_{\text{all}}(q,v)=
\begin{bmatrix}
b_L(q,v)\\
b_R(q,v)
\end{bmatrix}.
\]

Let

\[
\Sigma=\operatorname{diag}(s),
\]

with \(s_i=1\) for active stance rows and \(s_i=0\) for inactive rows.

Then

\[
\bar A = \Sigma A_{\text{all}},
\qquad
\bar b = \Sigma b_{\text{all}}.
\]

Define

\[
D_{\text{inactive}}=\operatorname{diag}(1-s).
\]

### 3.3 Constrained forward-dynamics KKT solve

Solve

\[
\begin{bmatrix}
M(q) & -\bar A(q)^\top\\
\bar A(q) & D_{\text{inactive}}
\end{bmatrix}
\begin{bmatrix}
a\\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
S^\top\tau-h(q,v)\\
-\bar b(q,v)
\end{bmatrix}.
\]

For active contact rows, \(D_{\text{inactive},ii}=0\), so the second block row enforces

\[
\bar A_i a + \bar b_i = 0.
\]

For inactive rows, \(\bar A_i=0\), \(\bar b_i=0\), and \(D_{\text{inactive},ii}=1\), so

\[
\lambda_i=0.
\]

Thus the inactive contact wrench is zero automatically.

### 3.4 acados dynamics

After the solve, use

\[
\dot q = T(q)v,
\]

\[
\dot v = a.
\]

So

\[
\dot x =
\begin{bmatrix}
T(q)v\\
a
\end{bmatrix}.
\]

In CasADi-style pseudocode:

```python
q, v = split_state(x)
tau = u

A_all = ca.vertcat(A_left(q), A_right(q))
b_all = ca.vertcat(b_left(q, v), b_right(q, v))

Sigma = ca.diag(s)
Abar = Sigma @ A_all
bbar = Sigma @ b_all
D_inactive = ca.diag(1.0 - s)

KKT = ca.vertcat(
    ca.horzcat(M(q), -Abar.T),
    ca.horzcat(Abar, D_inactive),
)

rhs = ca.vertcat(
    S.T @ tau - h_bias(q, v),
    -bbar,
)

sol = ca.solve(KKT, rhs)
a = sol[:nv]
lam = sol[nv:]

qdot = qdot_from_v(q, v)
vdot = a

model.f_expl_expr = ca.vertcat(qdot, vdot)
```

### 3.5 Constraints and costs

Use physical \(\lambda\) for contact constraints:

- \(F_z \ge 0\)
- friction cone or friction pyramid
- CoP rectangle
- contact wrench limits

Use \(\tau\) for torque limits and torque cost.

Do **not** include `ZeroAccel` in `model.con_h_expr`.

Recommended split:

```text
ZeroAccel:
    handled by KKT solve
    not in con_h_expr

ZeroWrench:
    inactive lambda rows are zero automatically

Friction / CoP / Fz >= 0:
    constraints on lambda

SwingZ:
    cost or soft inequality, not a physical contact KKT row

Torque limits:
    bounds or constraints on tau
```

---

## 4. Pipeline B: inverse-dynamics elimination in y = [a; f_c]

Use this if you want to stay closer to a whole-body MPC transcription with acceleration and contact wrench variables.

Instead of using \(\tau\) as input and solving forward dynamics, define

\[
y=
\begin{bmatrix}
a\\
f_c
\end{bmatrix},
\]

where \(a\) is generalized acceleration and \(f_c\) are contact wrenches.

Then stack the affine hard equalities as

\[
E(q,v,s)y+e(q,v,s)=0.
\]

These should include:

1. Active stance acceleration constraints.
2. Unactuated floating-base dynamics.

They should **not** include swing-foot tracking tasks as physical contact constraints.

### 4.1 Stance acceleration rows

For active stance contacts:

\[
J_c(q)a+
\dot J_c(q,v)v+
K_d e_v(q,v)+K_p e_x(q)=0.
\]

These rows are affine in \(y=[a;f_c]\) because they depend only on \(a\), not on \(f_c\).

### 4.2 Floating-base dynamics rows

The full rigid-body inverse dynamics are

\[
M(q)a+h(q,v)=S^\top\tau+J_c(q)^\top f_c.
\]

The floating-base rows are unactuated, so they must satisfy

\[
S_b\left(M(q)a+h(q,v)-J_c(q)^\top f_c\right)=0.
\]

These rows are affine in \(a\) and \(f_c\).

### 4.3 Thin elimination

Partition

\[
y=S_d y_d+S_f y_f,
\]

where \(y_d\) are dependent variables and \(y_f\) are free optimizer inputs.

Then

\[
E_d(q,v)y_d+E_f(q,v)y_f+e(q,v)=0.
\]

If \(E_d\) is square and nonsingular, solve

\[
y_d=-E_d(q,v)^{-1}\left(e(q,v)+E_f(q,v)y_f\right).
\]

Then reconstruct

\[
y(q,v,y_f)=S_dy_d+S_fy_f.
\]

Extract

\[
a = y_a,
\qquad
f_c = y_f^{\text{contact}},
\]

and compute actuator torques from the actuated rows:

\[
\tau = S_j\left(M(q)a+h(q,v)-J_c(q)^\top f_c\right).
\]

The acados dynamics are then

\[
\dot q=T(q)v,
\qquad
\dot v=a.
\]

### 4.4 Full-size weighted elimination

If changing dimensions by contact mode is inconvenient, use a full-size projected variable \(w_y\):

\[
y
=
\arg\min_y
\frac12(y-w_y)^\top W_y(y-w_y)
\quad
\text{s.t.}
\quad
E(q,v,s)y+e(q,v,s)=0.
\]

Then

\[
G_y = E W_y^{-1} E^\top + D_{\text{inactive}},
\]

\[
y_0=-W_y^{-1}E^\top G_y^{-1}e,
\]

\[
P_y=I-W_y^{-1}E^\top G_y^{-1}E,
\]

\[
y=y_0+P_yw_y.
\]

You must add a small positive definite cost directly on the raw input \(w_y\):

\[
\ell_{\text{reg}}=\frac12 w_y^\top R_w w_y,
\qquad
R_w\succ0.
\]

This is necessary because components of \(w_y\) in \(\ker(P_y)\) do not affect the reconstructed physical variables.

### 4.5 Why Pipeline B is often a good fit

Pipeline B is often closer to an existing whole-body controller because it keeps acceleration and contact wrench as the natural internal variables.

The optimizer input can be:

```text
thin form:
    y_free

full-size form:
    w_y
```

Inside the model, reconstruct:

```text
a
contact wrench f_c
torque tau
```

Then use:

```text
dynamics:
    qdot = T(q) v
    vdot = a

constraints:
    torque limits on tau
    friction / CoP / fz >= 0 on f_c
    swing clearance as soft inequality or cost

costs:
    tracking
    tau cost
    contact force cost
    raw input regularization
```

---

## 5. Why SwingZ should usually not be in the physical KKT

A swing-foot height task can be written at acceleration level:

\[
J_{z,\text{swing}}(q)a
+
\dot J_{z,\text{swing}}(q,v)v
+
K_d(\dot z-\dot z_{\text{ref}})
+
K_p(z-z_{\text{ref}})
-
\ddot z_{\text{ref}}
=0.
\]

This is mathematically affine in \(a\), so it can be included in a **task projection**.

However, it should not be included as a physical contact row in a constrained-EoM KKT solve. In the KKT system, every row of \(A\) creates a multiplier force through \(A^\top\lambda\). If you include a swing-foot height task there, the model invents an external force at the swing foot.

Recommended treatment:

```text
Stance-foot no-slip / no-acceleration:
    physical equality, eliminate exactly

Swing-foot height:
    tracking cost or soft inequality

Swing-foot clearance:
    z_swing >= z_min(t), preferably soft
```

Only include SwingZ in a projection if you explicitly interpret it as a task-space acceleration objective, not as a physical contact constraint.

---

## 6. Baumgarte / PD stabilization for stance drift

If you only enforce

\[
J_c a+\dot J_c v=0,
\]

you enforce zero contact acceleration but do not correct accumulated contact position or velocity error. That can allow stance-foot drift over the horizon.

Use

\[
J_c a
+
\dot J_c v
+
K_d e_v
+
K_p e_x
=0.
\]

Therefore

\[
b(q,v)=\dot J_c v+K_d e_v+K_p e_x.
\]

Then the residual check is

\[
r_c = J_c(q)a+b(q,v).
\]

This should be near numerical precision for active stance rows.

Be careful with signs. If your implementation defines `zero_accel_b` as the negative RHS, then the sign of \(a_0\) or the KKT right-hand side flips.

The safest test is always:

```text
compute reconstructed a
compute residual r = A a + b
verify ||r||_inf is tiny
```

---

## 7. Raw-input regularization is mandatory for full-size projected variables

If you use

\[
a=a_0+Pw_a,
\]

or

\[
y=y_0+P_yw_y,
\]

then \(P\) or \(P_y\) is rank-deficient whenever constraints are active.

That means some raw input directions do not affect:

- acceleration,
- state dynamics,
- physical forces,
- physical torques,
- costs written only on reconstructed physical variables.

Therefore, a cost on the projected variable is not enough.

Bad:

\[
\ell \supset \|a\|_Q^2.
\]

Good:

\[
\ell \supset \|a\|_Q^2 + \epsilon\|w_a\|^2.
\]

For the inverse-dynamics projection:

\[
\ell \supset \|\tau\|_R^2+\|f_c\|_Q^2+\epsilon\|w_y\|^2.
\]

The regularization must be directly on the raw acados input.

---

## 8. `ca.solve` and code-generation risk

Putting a solve inside the dynamics is mathematically clean, but the generated CasADi code can grow.

The explicit formulation is:

```python
z = ca.solve(KKT, rhs)
a = z[:nv]
lam = z[nv:]
model.f_expl_expr = ca.vertcat(qdot, a)
```

This works with explicit integration, but acados/CasADi must differentiate through the solve.

The cost depends on:

- KKT size,
- number of shooting nodes,
- integrator type,
- RK stages,
- whether exact Hessians are used,
- expression complexity of \(M(q)\), \(J(q)\), and \(h(q,v)\).

Recommended prototype sequence:

```text
1. Build only the dynamics expression with the solve.
2. Generate the external function code.
3. Check generated .c file size.
4. Compile it.
5. Time f_expl and its Jacobian.
6. Only then add costs and constraints.
```

A smaller explicit acceleration projection might solve a 12x12 system.

A constrained forward-dynamics KKT might solve something like a 45x45 system.

The KKT version is more physical but may generate larger code. Measure both.

---

## 9. Alternative: implicit DAE formulation

Instead of explicitly solving the KKT inside `f_expl_expr`, you can formulate an implicit DAE.

Use algebraic variables

\[
z=
\begin{bmatrix}
a\\
\lambda
\end{bmatrix}.
\]

Then define implicit dynamics:

\[
\dot q - T(q)v = 0,
\]

\[
\dot v - a = 0,
\]

\[
M(q)a+h(q,v)-S^\top\tau-J_c(q)^\top\lambda=0,
\]

\[
J_c(q)a+b(q,v)=0.
\]

This avoids explicitly writing `ca.solve(KKT, rhs)` in the dynamics expression, but moves the solve into the implicit integrator machinery.

Tradeoffs:

```text
Explicit solve:
    simpler ODE interface
    may generate large code
    easier to expose a and lambda directly

Implicit DAE:
    potentially cleaner symbolic structure
    uses implicit integrator
    may be more expensive per integration step
    algebraic variable handling must be tested carefully
```

---

## 10. Recommended final architecture

Given your current issue and desire to stay close to a whole-body controller, the best candidate is usually **Pipeline B: inverse-dynamics elimination**.

Recommended structure:

```text
Offline:
    1. Derive ZeroAccel as A(q) a + b(q,v) = 0.
    2. Include Baumgarte / PD terms in b.
    3. Derive floating-base dynamics rows.
    4. Stack equalities E(q,v,s) y + e(q,v,s) = 0 with y = [a; f_c].
    5. Check active-row rank over representative walking states.

Model generation:
    6. Define acados input as y_free or full-size w_y.
    7. Reconstruct y = [a; f_c] inside the model.
    8. Compute tau from actuated inverse dynamics.
    9. Set qdot = T(q) v and vdot = a.
    10. Remove ZeroAccel from con_h_expr.
    11. Add torque limits, friction, CoP, unilateral contact, joint limits, and swing clearance.
    12. Add raw input regularization.

Runtime:
    13. Set contact masks and references stage-wise via parameters.
    14. Set active/inactive bounds for remaining constraints.
    15. Solve acados.
    16. Recover physical a, f_c, and tau.
    17. Validate residuals and conditioning.
```

---

## 11. Validation checklist

At every shooting node, evaluate the following after solving.

### 11.1 Contact acceleration residual

\[
r_c = J_c(q)a+b_c(q,v).
\]

Expected:

\[
\|r_c\|_\infty \approx 0
\]

for active stance rows.

### 11.2 Floating-base dynamics residual

For inverse-dynamics elimination:

\[
r_b = S_b\left(M(q)a+h(q,v)-J_c(q)^\top f_c\right).
\]

Expected:

\[
\|r_b\|_\infty \approx 0.
\]

### 11.3 Forward-dynamics residual

For torque-input constrained forward dynamics:

\[
r_{\text{dyn}} = M(q)a+h(q,v)-S^\top\tau-J_c(q)^\top\lambda.
\]

Expected:

\[
\|r_{\text{dyn}}\|_\infty \approx 0.
\]

### 11.4 Contact force validity

Check:

\[
F_z \ge 0,
\]

\[
|F_x| \le \mu F_z,
\]

\[
|F_y| \le \mu F_z,
\]

and CoP rectangle constraints.

### 11.5 Rank and conditioning

For the contact-only system:

\[
\sigma_{\min}(J_{\text{active}})
\]

should stay away from zero.

For inverse-dynamics elimination:

\[
\sigma_{\min}(E_{\text{active}})
\]

or the condition number of the eliminated block should be monitored.

For forward-dynamics KKT:

\[
\kappa
\left(
\begin{bmatrix}
M & -\bar A^\top\\
\bar A & D_{\text{inactive}}
\end{bmatrix}
\right)
\]

should be monitored.

### 11.6 QP conditioning

After reformulation, use acados QP diagnostics if available in your interface, and compare before/after:

```python
solver.qp_diagnostics("FULL_HESSIAN")
solver.qp_diagnostics("PROJECTED_HESSIAN")
```

The important qualitative change is that `ZeroAccel` should no longer appear as hard equality rows in the QP.

---

## 12. Bottom line

Your understanding of the weighted projection math is correct.

The refined conclusion is:

```text
Pure acceleration projection:
    exact ZeroAccel elimination
    but kinematic unless dynamics are included elsewhere

Torque-input constrained forward dynamics:
    physically faithful
    input is tau
    lambda is physical contact wrench
    no SwingZ in the physical KKT

Inverse-dynamics elimination:
    closest to a qdd + contact-wrench whole-body formulation
    eliminates ZeroAccel and floating-base dynamics before acados forms the QP
    reconstructs physical acceleration, contact wrench, and torque
    likely best match for your existing controller

Full-size projected inputs:
    require raw-input regularization

Baumgarte / PD terms:
    should be folded into b(q,v)

ca.solve/codegen:
    must be prototyped and measured early
```

The main goal is:

\[
\boxed{\text{Do not let active ZeroAccel rows enter HPIPM as hard path equalities.}}
\]

Instead, reconstruct physical accelerations and forces inside the model so those equalities are satisfied by construction.

---

## References

- acados Python interface documentation: `model.f_expl_expr`, `model.f_impl_expr`, `model.con_h_expr`, constraints, parameters, and solver setters.  
  https://docs.acados.org/python_interface/index.html

- OCS2 optimal control modules documentation: state-input equality constraints are handled through a projection method with a full-row-rank input Jacobian assumption.  
  https://leggedrobotics.github.io/ocs2/optimal_control_modules.html
