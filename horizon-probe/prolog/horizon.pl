:- dynamic model/2, item/3, answer/6, axis_p/4, item_div/5, level/2, token_stat/5,
           token_rate/4, boundary_token/2, lexicon/2, dbn_unit/4, threshold/2.
:- discontiguous threshold/2.

side(legal, M, I, legal) :- axis_p(M, I, PL, _), PL >= 0.5.
side(legal, M, I, illegal) :- axis_p(M, I, PL, _), PL < 0.5.
side(banned, M, I, banned) :- axis_p(M, I, _, PB), PB >= 0.5.
side(banned, M, I, not_banned) :- axis_p(M, I, _, PB), PB < 0.5.

quadrant(M, I, Q) :-
    answer(M, I, PA, PB, PC, PD),
    max_member(_-Q, [PA-a, PB-b, PC-c, PD-d]).

boundary_item(I) :-
    item_div(I, JSD, LogBF, _, _),
    threshold(item_jsd, T), JSD >= T,
    threshold(log_bf, B), LogBF >= B.

divergence_axis(I, legal) :- item_div(I, _, _, JL, JB), JL >= JB.
divergence_axis(I, banned) :- item_div(I, _, _, JL, JB), JB > JL.

split(I, Axis, M1, M2) :-
    member(Axis, [legal, banned]),
    side(Axis, M1, I, S1), side(Axis, M2, I, S2),
    S1 \== S2, M1 @< M2.

universal(I, Q) :-
    item(I, _, _),
    once(quadrant(_, I, Q)),
    forall(model(M, _), quadrant(M, I, Q)).

universal_side(Axis, I, S) :-
    item(I, _, _),
    member(Axis, [legal, banned]),
    once(side(Axis, _, I, S)),
    forall(model(M, _), side(Axis, M, I, S)).

tier_pivot(I, Axis, R) :-
    item(I, _, _),
    member(Axis, [legal, banned]),
    setof(Rk, M^model(M, Rk), Ranks),
    member(R, Ranks),
    findall(S, (model(M, Rk), Rk >= R, side(Axis, M, I, S)), Hi), Hi \== [],
    findall(S, (model(M, Rk), Rk < R, side(Axis, M, I, S)), Lo), Lo \== [],
    length(Hi, NH), length(Lo, NL), NH + NL >= 3,
    sort(Hi, [SH]), sort(Lo, [SL]), SH \== SL.

conflation(M, I) :- item(I, _, b), quadrant(M, I, c).

conflation_rate(M, K, N) :-
    model(M, _),
    aggregate_all(count, (item(I, _, b), answer(M, I, _, _, _, _)), N),
    aggregate_all(count, conflation(M, _), K).

anchor_hit(M, I) :- item(I, _, E), E \== none, quadrant(M, I, E).

anchor_accuracy(M, K, N) :-
    model(M, _),
    aggregate_all(count, (item(I, _, E), E \== none, answer(M, I, _, _, _, _)), N),
    aggregate_all(count, anchor_hit(M, _), K).

horizon(M, Kind, Axis, K, N) :-
    model(M, _),
    setof(Kd, I^E^item(I, Kd, E), Kinds),
    member(Kind, Kinds),
    member(Axis-Pos, [legal-legal, banned-banned]),
    aggregate_all(count, (item(I, Kind, _), answer(M, I, _, _, _, _)), N),
    N > 0,
    aggregate_all(count, (item(I, Kind, _), side(Axis, M, I, Pos)), K).

persistent_token(T) :- boundary_token(L1, T), L2 is L1 + 1, boundary_token(L2, T).

last_level(L) :- aggregate_all(max(X), level(X, _), L), number(L).

fixed_point_token(T) :-
    last_level(L), L > 0, P is L - 1,
    boundary_token(L, T), boundary_token(P, T).

axis_token(L, T, Sense) :- boundary_token(L, T), lexicon(T, Sense).

emit(Tag, Args) :-
    atomic_list_concat(Args, '\t', Row),
    format("~w\t~w~n", [Tag, Row]).

report :-
    forall(boundary_item(I), (divergence_axis(I, A), item_div(I, J, B, _, _), emit(boundary_item, [I, A, J, B]))),
    forall(split(I, Axis, M1, M2), emit(split, [I, Axis, M1, M2])),
    forall(universal(I, Q), emit(universal, [I, Q])),
    forall(universal_side(Axis, I, S), emit(universal_side, [Axis, I, S])),
    forall(tier_pivot(I, Axis, R), emit(tier_pivot, [I, Axis, R])),
    forall(conflation_rate(M, K, N), emit(conflation, [M, K, N])),
    forall(anchor_accuracy(M, K, N), emit(anchor, [M, K, N])),
    forall(horizon(M, Kind, Axis, K, N), emit(horizon, [M, Kind, Axis, K, N])),
    forall(persistent_token(T), emit(persistent_token, [T])),
    forall(fixed_point_token(T), emit(fixed_point_token, [T])),
    forall(axis_token(L, T, S), emit(axis_token, [L, T, S])).
