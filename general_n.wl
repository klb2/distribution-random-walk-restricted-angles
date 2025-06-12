(* ::Package:: *)

(* ::Title:: *)
(* General Number of Steps *)


(* ::Chapter:: *)
(* Condition for Unique Angles in Support *)


(* ::Section:: *)
(* Lemma 1 *)


suppPart[x_, n_, a_] := (n-1)*Sin[a] - Sqrt[1-(x-(n-1)*Cos[a])^2]
derivSupp[x_, n_, a_] = D[suppPart[x, n, a], x]
FullSimplify[derivSupp[n*Cos[a], n, a], Assumptions -> {n >= 3, 0<a<=Pi/2}]
TrigReduce[Tan[x]^2]
Simplify[
  Solve[(1 - Cos[2 x])/(1 + Cos[2 x]) == n/(n - 2), x, Reals], 
Assumptions -> {n >= 3}]



(* ::Section:: *)
(* Corollary 3 *)


Limit[0.5*ArcCos[-1/(n - 1)], n -> Infinity]



(* ::Chapter:: *)
(* Distributions of $R_N$ and $\theta_N$ *)


(* ::Section:: *)
(* PDF Transformation *)


(* ::Text:: *)
(* We need to determine the Jacobian determinant of the inverse transformation *)


Inverse[
  {{Cos[t], -r*Sin[t], -Sin[p]}, {Sin[t], r*Cos[t], Cos[p]}, {0, 0, 1}}
]
FullSimplify[%]
Det[
  {{Cos[t], Sin[t], Sin[p - t]}, {-(Sin[t]/r), Cos[t]/ r, -(Cos[p - t]/r)}, {0, 0, 1}}
]



(* ::Section:: *)
(* Approximation of the Angle *)


phiN[x_, y_, r_, t_] := ArcTan[(r*Sin[x] + Sin[y])/(r*Cos[x] + Cos[y])] - t
derivPhiN[x_, y_, r_, t_] = ImplicitD[phiN[x, y, r, t] == 0, y, x]



(* ::Text:: *)
(* We use a Taylor expansion (linear) around the point $(\theta_N, \theta_N)$ *)


Simplify[t + derivPhiN[t, t, r, t]*(x-t)]



(* ::Section:: *)
(* Illustration *)


Manipulate[
  ContourPlot[{
    ArcTan[(r*Cos[x] + Cos[y]), (r*Sin[x] + Sin[y])] == t, 
    y == t - r*(x - t)
  },
  {x, -Pi/2, Pi/2}, {y, -Pi/2, Pi/2}],
{r, 1, 5}, {t, 0, Pi/2}]
