# Two definitions of four words: legal, wants, shared borders, mutual success

Kaggriculture (team 4H). Neighboring autonomous farmers share borders and one market. The same four words get two definitions:
- **A:** rules only.
- **B:** a Nash equilibrium between neighbors.

| Word | A: rules only | B: Nash equilibrium between neighbors |
|---|---|---|
| Legal | Any move the game engine accepts | Any move the engine accepts, plus commitments a neighbor can enforce (the $80 underwriting) |
| Wants | Maximize own 5-day forecast profit; the neighbor is noise | Maximize own profit given the neighbor's committed plan; nobody gains by deviating |
| Shared borders | A collision zone: both tiles sell into the same market depth | A contract surface: one place to trade a commitment |
| Mutual success | Luck: neighbors differ in about 30% of seasons | The equilibrium itself: +$120 each, every season |

## Payoffs per border per season

These are illustrative: the smallest matrix that reproduces the $80 / $120 / $120 figures. They are not fitted from replays.

| | B plants X (forecast peak) | B plants Y |
|---|---|---|
| **A plants X** | $100 / $100 (both dump into one market) | $300 / $140 |
| **A plants Y** | $140 / $300 | $120 / $120 |

**Definition A (no enforcement).** The only stable play is mixed:
- Each farmer plants X with probability 9/11, for about $136 each.
- Both dump into the same market (100/100) in 81/121 ≈ 67% of seasons.
- The neighbors plant different crops in 36/121 ≈ 30% of seasons.

**Definition B (enforceable underwriting).**
- A pays B $80 to plant Y. Each ends at $220: +$120 and +$120 over the dump outcome.
- The $80 is the Nash bargaining split of the $240 surplus over the dump outcome. It maximizes (u_A − 100)(u_B − 100).
- Each farmer reinvests $40 of their $120 into the next border's underwriting, so contracts double each season (1 → 2 → 4 → 8 → 16 → 24).

## Farm level (4×4 farms, 24 shared borders, every contract honored)

| Season | Rules-only cumulative surplus | Underwriting cumulative surplus | Ratio |
|---|---|---|---|
| 1 | $1,745 | $1,913 | 1.10 |
| 5 | $8,727 | $13,913 | 1.59 |
| 6 | $10,473 | $19,673 | 1.88 |
| 12 | $20,945 | $54,233 | 2.59 |

**Belief uncertainty about a neighbor's move.**
- Without a contract it is H(9/11) = 0.68 bits per border.
- Under a contract honored with probability q it is H(q).
- A contract honored less than about 82% of the time (9 in 11) therefore leaves the neighbor harder to predict than no contract at all.
- Uncertainty reaches 0 bits only when q = 1. At that point asking the neighbor carries no information, so asking becomes ordering: the peer turns into a subagent.

## The A/C split across model sizes

Large models answered "A) Legal, not banned" and smaller models answered "C) Illegal and banned". The simplest explanation is word-sense resolution:
- Larger models read "legal" in the in-game sense from context.
- Smaller models fall back to the real-world sense and the cautious answer.

**Control:** disambiguate the word ("permitted by the game engine" vs "lawful"). If the gap between sizes closes, it was the word that switched, not the agent. `horizon-probe/` builds this test across Opus, Sonnet and the Gemma 4 sizes.

Interactive version: `legal-two-definitions.html` (open in a browser).
