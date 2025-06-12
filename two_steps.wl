(* ::Package:: *)

(* ::Title:: *)
(*Two Steps*)


(* ::Chapter:: *)
(*Radius given Angle*)


Manipulate[
  Plot[{
    Sqrt[2*(1+Cos[2*(x-t)])],
    Sqrt[2*(1+Cos[2*(t-a)])],
    r
  },{x,-a,a},
  PlotRange->{{-a,a},{0,2}},
  Epilog->{
    Line[{{2*t-a,0},{2*t-a,2}}],
    Line[{{-ArcCos[r^2/2-1]/2+t,0},{-ArcCos[r^2/2-1]/2+t,2}}]
  }],
{t,0,a},{a,0.01,1.5},{r,0,2}]


cdfCond[r_,t_,a_]:=Piecewise[{{1-ArcCos[r^2/2-1]/(2*(a-t)),Sqrt[2*(1+Cos[2*(t-a)])]<r<2},{1,r>=2}}]
radInner[t_,a_]:=Sqrt[2*(1+Cos[2*(t-a)])]


Manipulate[Plot[cdfCond[r,t,a],{r,0,2.1}, PlotRange->{0,1}],{t,0,a},{a,0,Pi/2}]


(* ::Chapter:: *)
(*Joint Distribution*)


D[(1-(ArcCos[r^2/2-1]/2)/(a-t))*(1/a*(1-t/a)),r]


Simplify[%,Assumptions->{a>0,r>0,t>0}]


(* ::Chapter:: *)
(*Marginal Distribution - Radius*)


Integrate[cdfCond[r,-t,a]*1/a*(1+t/a),t,Assumptions->{a>0,x>0,r<2}]
