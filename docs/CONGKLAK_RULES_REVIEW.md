# Congklak Sequential Rules Review Draft

Review date: 17 October 2026. This is the complete human-readable description of
the current `id-betawi-relay7-v0` game, ready for discussion. It records the
sequential opening selected by the user. It does not change the engine, saved
games, or the frozen machine-readable rule contract.

The scope is seven pits per player, seven starting seeds per pit, sequential
play, and one round. Some edge cases below are existing engine conventions that
still need confirmation as the intended house rules. This is not a claim that
every Betawi or Dakon tradition uses identical rules.

## Board and starting position

Two players, A and B, sit opposite each other. Each owns seven small pits and
one large store. There are 98 seeds: seven in each of the fourteen small pits.
Both stores start empty.

The browser defaults to the user's preferred mirrored presentation: the store
is on each player's left and sowing runs right to left from that player's seat.
The Ministry description and frozen engine contract use the equivalent
right-store layout. The Board layout selector can show either. This reflection
changes displayed coordinates, not the rules or saved positions.

Seed counts are public and exact. Players may count seeds and plan their choices.

The browser board uses these labels, viewed from A's side:

```text
                     Player B
              B7 B6 B5 B4 B3 B2 B1
    A store                            B store
              A1 A2 A3 A4 A5 A6 A7
                     Player A
```

Opposite pairs are A1/B7, A2/B6, A3/B5, A4/B4, A5/B3, A6/B2 and A7/B1.
Pit numbers identify locations, not separate moves made by both players.

## Opening

Choose who starts before playing. The current app assigns the default opening
to A; this is the app's starting convention, not a claim that a particular person
must always start. Only the selected starter moves. The two players do not sow
simultaneously.

The starter obeys the same sowing, relay, bonus and capture rules as later play.
There is no special opening limit on the number of extra choices.

## Choosing and sowing

When it is your turn to choose, select any nonempty small pit on your own side
and pick up all its seeds. You cannot select an empty pit, an opponent's pit, or
either store as your initial choice.

Place the seeds one at a time into successive locations, beginning immediately
after the selected pit. Include your own store and skip your opponent's store.
A full lap may return to the pit you originally emptied; it receives a seed
normally if you still have seeds in hand.

To avoid ambiguous clockwise descriptions, the default left-store display's
fixed route is:

```text
A7 -> A6 -> A5 -> A4 -> A3 -> A2 -> A1 -> A store
   -> B7 -> B6 -> B5 -> B4 -> B3 -> B2 -> B1 -> B store -> A7
```

A skips B's store; B skips A's store. Each player follows this same route from
their chosen location. Seeds already in a store are never picked up or captured.

## What happens when the hand becomes empty

"Empty pit" means the pit contained no seeds immediately before the last seed
was deposited. Resolve the following cases in order:

| Last seed lands in | What happens | Who chooses next |
| --- | --- | --- |
| Your own store | Leave the seeds in the store. You receive another choice from your own nonempty pits. | You, unless your row is empty or the round has ended. |
| A small pit that already contained seeds, on either side | Pick up all seeds now in that pit, including the seed just deposited, and continue sowing from the next location. | Nobody chooses yet; the same sowing action continues. |
| An empty pit on your side, with seeds in its opposite pit | Transfer the landing seed and all seeds from the opposite pit into your store. Both small pits become empty. | The opponent, subject to an empty-row pass. |
| An empty pit on your side, with an empty opposite pit | Leave the landing seed in its pit. Capture nothing. | The opponent, subject to an empty-row pass. |
| An empty pit on the opponent's side | Leave the landing seed there. Capture nothing. | The opponent, subject to an empty-row pass. |

A relay is compulsory and can repeat many times. Relaying from an opponent's
occupied pit does not make it the opponent's turn.

Only the last seed landing in your own store grants another pit choice.
Passing through your store while still holding seeds does not grant an immediate
choice. A capture adds seeds to your store but does not itself grant a bonus.

Under the current engine convention, capture eligibility is unrestricted: no
completed lap or previous visit to the opponent's row is required.

### Capture handoff verification

Checked on 17 October 2026: the Ministry's Betawi description says that the
player stops and takes the opposite seeds. Its wording includes "maka ia
berhenti dan mengambil". This supports ending the turn after capture. Moving
captured seeds into the store does not count as sowing the last seed there and
does not create an own-store bonus.

The supplied Hasya TV tutorial's Indonesian transcript also explicitly says to
switch players after taking the opposite seeds into the player's store. It
demonstrates that handoff again during play. This verifies agreement on capture
handoff, without establishing every other rule in that tutorial as our profile.

## Empty rows and passing

After sowing and any capture are complete, first check whether the round has
ended. Otherwise, a player who is due to choose but has no seeds in their own
small pits must pass to the other player.

This also applies after an own-store bonus: if that player has no filled pit to
choose, the other player moves. If an ordinary turn ends but the opponent's row
is empty, the opponent passes and the original player chooses again.

An empty row does not end the round by itself. A row may receive seeds during
later sowing and become playable again. A player with a filled own pit cannot
voluntarily pass.

## End of the round and winner

The round ends only when both rows of small pits are empty, after a complete
sowing action and any capture. All 98 seeds are then in the stores.

The player with more stored seeds wins. A final score of 49-49 is a draw.
There is no automatic sweep of the remaining seeds when just one row empties.
There are no refills, closed pits, protected pits or additional rounds in this
profile. Nothing removes seeds from a store during the round.

A player who has already stored at least 50 seeds has secured the eventual win.
This mathematical certainty does not, by itself, end play early. The engine's
intermediate evaluation is an estimate and is not another scoring rule.

## Worked examples

- On the untouched starting board in the default left-store view, A chooses A7
  and sows its seven seeds into A6, A5, A4, A3, A2, A1 and A's store. A has one
  stored seed and chooses again.
- If your last seed lands in a pit that already held three seeds, pick up all
  four and continue. The same rule applies on the opponent's row.
- If your last seed enters an empty own pit opposite five seeds, capture six
  into your store: those five plus your landing seed. Your turn then ends.
- If that opposite pit is empty, the landing seed stays in your pit and your
  turn ends. No capture occurs.
- If the opponent's row is empty when your turn ends, the opponent passes.
  Your next choice is legal because of that pass, not a new own-store bonus.

## Details still to confirm as our intended house rules

These are implemented today, but the brief cultural sources do not fully settle
them. Keep them explicit during the review rather than treating silence as proof.

| Detail | Current engine behavior |
| --- | --- |
| Landing seed in an own-empty capture | Captured together with the opposite seeds. |
| Empty opposite pit | No capture; leave the landing seed and end the turn. |
| Capture before completing a lap | Allowed; no first-lap restriction. |
| A player has an empty row | Pass and continue; do not end the round or sweep remaining seeds. |

The linked video's transcript has now been checked for capture handoff and key
rule differences. It agrees that capture ends the turn, but uses six starting
seeds per pit, ends a turn on an own-store landing, and ends the game when one
row empties, moving the other row's remaining seeds into its owner's store.
Those choices differ from our selected sequential Ministry profile. This was a
transcript review, not a complete visual audit of seed handling. Record any
proposed changes separately before changing the engine or a saved game's rules.

## Known consequence of the current profile

The following one-based pit choices in the default left-store view form a
verified bonus-choice chain from the starting position:

```text
7, 1, 4, 7, 2, 2, 5, 4, 1, 6, 4, 7, 1, 2
```

The starter banks 51 seeds before the opponent can choose. Both the native
simulator and the independent literal interpreter agree. This establishes a
winning strategy for the starter under these exact rules, not a guarantee that
every opening move wins. Changing the opening, bonus rules, seed count or match
format would require checking the consequence again.

The saved certificate retains its original right-store labels:
`1,7,4,1,6,6,3,4,7,2,4,1,7,6`. Each label maps by `new = 8 - old` when mirrored.
The certificate, engine indices and imported/exported JSON arrays are unchanged.

## References and implementation boundary

- Selected sequential description: [Ministry of Education and Culture, Congklak Betawi](https://kebudayaan.kemdikbud.go.id/ditwdb/congklak-betawi/).
- Local source audit and its limits: [Betawi rules audit](../build/betawi-rules-audit-v1/rules-audit.json).
- Frozen executable contract: [id-betawi-relay7-v0](../rules/id-betawi-relay7-v0.json).
- Video supplied for further comparison: [Hasya TV tutorial](https://www.youtube.com/watch?v=6eO3N5pWpKM).

Stable positions contain the fourteen pit counts, two store counts, and player
to choose. They always total 98. Mid-sowing positions with seeds in hand are not
accepted as stable imports. Software timeouts or interrupted analyses do not
declare a game result.
