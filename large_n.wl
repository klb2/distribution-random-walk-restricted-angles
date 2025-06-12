(* ::Package:: *)

(* ::Title:: *)
(*Approximation for Large N*)


(* ::Chapter:: *)
(*CLT Parameters*)


(* ::Text:: *)
(*In the following, we calculate all parameters for the CLT (mean, variance, and covariance of the joint normal distribution).*)


(* ::Section:: *)
(*Mean Values*)


Integrate[1/(2*a) *Cos[t],{t,-a,a}]
Integrate[1/(2*a)*Sin[t],{t,-a,a}]


(* ::Section:: *)
(*Variances*)


Integrate[1/(2*a)*Cos[t]^2,{t,-a,a}]-(Sin[a]/a)^2
Integrate[1/(2*a)*Sin[t]^2,{t,-a,a}]


(* ::Section:: *)
(*Covariance*)


Integrate[1/(2*a)*(Cos[t]-Sin[a]/a)*Sin[t],{t,-a,a}]


(* ::Chapter:: *)
(*Derivative of CDF*)


(* ::Section:: *)
(*Derivative of Argument of Ratio CDF*)


FullSimplify[D[(n*m*w)/(Sqrt[n*(w^2*x^2+y^2)]),w],Assumptions->{n>=2}]


(* ::Section:: *)
(*Derivative of Argument of Angle CDF*)


D[Tan[t],t]
